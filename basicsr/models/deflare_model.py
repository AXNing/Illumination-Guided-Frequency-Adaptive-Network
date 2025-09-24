
from collections import OrderedDict
from os import path as osp

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.flare_util import blend_light_source,mkdir,predict_flare_from_6_channel,predict_flare_from_3_channel
from kornia.metrics import psnr,ssim
from basicsr.metrics import calculate_metric
import torch
from tqdm import tqdm
import torchvision
from torchstat import stat
#from fvcore.nn import FlopCountAnalysis, parameter_count_table
from basicsr.losses.cut_loss import CutLoss
from basicsr.losses.fft_loss import get_Fre
from clip_loss import L_clip_MSE
import torch.nn as nn
import os
import random

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_


def get_batchnorm_layer(opts):
    if opts.norm_layer == "batch":
        norm_layer = nn.BatchNorm2d
    elif opts.layer == "spectral_instance":
        norm_layer = nn.InstanceNorm2d
    else:
        print("not implemented")
        exit()
    return norm_layer

def get_conv2d_layer(in_c, out_c, k, s, p=0, dilation=1, groups=1):
    return nn.Conv2d(in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=s,
                    padding=p,dilation=dilation, groups=groups)

def get_deconv2d_layer(in_c, out_c, k=1, s=1, p=1):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=k,
            stride=s,
            padding=p
        )
    )
class Decom(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(in_c=3, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=4, k=3, s=1, p=1),
            nn.ReLU()
        )

    def forward(self, input):
        output = self.decom(input)
        # R = output[:, 0:3, :, :]
        # L = output[:, 3:4, :, :]
        return output

def aux_load_initialize(model, decom_model_path):
    if os.path.exists(decom_model_path):
        checkpoint_Decom_low = torch.load(decom_model_path)
        model.load_state_dict(checkpoint_Decom_low['state_dict']['model_R'])
        # to freeze the params of Decomposition Model
        for param in model.parameters():
            param.requires_grad = False
        return model
    else:
        print("pretrained Initialize Model does not exist, check ---> %s " % decom_model_path)
        exit()

@MODEL_REGISTRY.register()       
class DeflareModel(SRModel):

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.output_ch=self.opt['network_g']['output_ch']
        if 'multi_stage' in self.opt['network_g']:
            self.multi_stage=self.opt['network_g']['multi_stage']
        else:
            self.multi_stage=1
        print("Output channel is:", self.output_ch)
        print("Network contains",self.multi_stage,"stages.")
        self.Decom_l = Decom().cuda()
        self.Decom_l = aux_load_initialize(self.Decom_l,self.opt['pretrain_decomnet_low'])
        self.Decom_l.eval()
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()
        

        # define losses
        self.l1_pix = build_loss(train_opt['l1_opt']).to(self.device)
        self.l_perceptual = build_loss(train_opt['perceptual']).to(self.device)
        self.l_cut = CutLoss()
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.l_fre = get_Fre()
        self.l_clip = L_clip_MSE() 
    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)
        if 'flare' in data:
            self.flare = data['flare'].to(self.device)
            self.gamma = data['gamma'].to(self.device)
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)
    
    def optimize_parameters(self, current_iter):
        with torch.no_grad():

            retinex = self.Decom_l(self.lq)



        self.optimizer_g.zero_grad()

        
        self.out_put = self.net_g(self.lq,retinex)



        if self.output_ch==6:
            self.deflare,self.flare_hat,self.merge_hat=predict_flare_from_6_channel(self.out_put,self.gamma)
        elif self.output_ch==3:
            self.mask=torch.zeros_like(self.lq).cuda() # Comment this line if you want to use the mask
            self.deflare,self.flare_hat=predict_flare_from_3_channel(self.out_put,self.mask,self.lq,self.flare,self.lq,self.gamma)        
        else:
            assert False, "Error! Output channel should be defined as 3 or 6."
        


        l_total = 0
        loss_dict = OrderedDict()
        # l1 loss
        l1_flare = self.l1_pix(self.flare_hat, self.flare)*2
        l1_base = self.l1_pix(self.deflare, self.gt)*2



        #l1=l1_flare+l1_base
        l1=l1_base
        if self.output_ch==6:
            l1_recons= self.l1_pix(self.merge_hat, self.lq)
            loss_dict['l1_recons']=l1_recons*2
            l1+=l1_recons*2
        l_total += l1






        loss_clip = (self.l_clip(self.deflare,self.gt))*1.2
        loss_clip_flare = (self.l_clip(self.flare_hat,self.flare))*1.2
        l_total +=loss_clip
        loss_dict['loss_clip']=loss_clip

        l_total +=loss_clip_flare
        loss_dict['loss_clip_flare']=loss_clip_flare

        gt_fre1,gt_fre2 = self.l_fre(self.gt)
        de_fre1,de_fre2 = self.l_fre(self.deflare)
        loss_fre1 = self.l1_pix(de_fre1,gt_fre1)
        loss_fre2 = self.l1_pix(de_fre2,gt_fre2)
        fre_loss = (loss_fre1 + loss_fre2)
        l_total +=fre_loss
        loss_dict['fre_loss']=fre_loss




        loss_dict['l1_flare']=l1_flare


        loss_dict['l1_base']=l1_base
        loss_dict['l1'] = l1

        # perceptual loss
        l_vgg_flare = self.l_perceptual(self.flare_hat, self.flare)
        l_vgg_base = self.l_perceptual(self.deflare, self.gt)
        l_vgg= l_vgg_base +l_vgg_flare
        l_total += l_vgg
        loss_dict['l_vgg'] = l_vgg
        loss_dict['l_vgg_base'] = l_vgg_base
        loss_dict['l_vgg_flare'] = l_vgg_flare

        l_total.backward()
        self.optimizer_g.step()
        
        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.retinex = self.Decom_l(self.lq)
                self.output = self.net_g_ema(self.lq,self.retinex)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
        if self.output_ch==6:
            self.deflare,self.flare_hat,self.merge_hat=predict_flare_from_6_channel(self.output,self.gamma)
        elif self.output_ch==3:
            self.mask=torch.zeros_like(self.lq).cuda() # Comment this line if you want to use the mask
            self.deflare,self.flare_hat=predict_flare_from_3_channel(self.output,self.mask,self.gt,self.flare,self.lq,self.gamma)        
        else:
            assert False, "Error! Output channel should be defined as 3 or 6."
        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            img_name='deflare_'+str(idx).zfill(5)+'_'
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        if self.output_ch==3:
            self.blend= blend_light_source(self.lq, self.deflare, 0.97)
            out_dict['result']= self.blend.detach().cpu()
        elif self.output_ch ==6:
            out_dict['result']= self.deflare.detach().cpu()
        out_dict['flare']=self.flare_hat.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict


