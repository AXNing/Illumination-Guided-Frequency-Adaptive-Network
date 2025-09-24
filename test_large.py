import glob
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageChops
from basicsr.data.flare7k_dataset import Flare_Image_Loader,RandomGammaCorrection
from basicsr.archs.uformer_arch import Uformer
import argparse
from basicsr.archs.unet_arch import U_Net
from basicsr.archs.SpaFormer_arch import SpA_former
from basicsr.utils.flare_util import blend_light_source,get_args_from_json,save_args_to_json,mkdir,predict_flare_from_6_channel,predict_flare_from_3_channel
from torch.distributions import Normal
import torchvision.transforms as transforms
import os
from torchstat import stat
from thop import profile
from heatmap import tensor_to_heatmap
import torch.nn as nn
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import math

parser = argparse.ArgumentParser()
parser.add_argument('--input',type=str,default=None)
parser.add_argument('--output',type=str,default=None)
parser.add_argument('--model_type',type=str,default='Uformer')
parser.add_argument('--model_path',type=str,default='')
parser.add_argument('--output_ch',type=int,default=6)
parser.add_argument('--flare7kpp', action='store_const', const=True, default=False) #use flare7kpp's inference method and output the light source directly.

args = parser.parse_args()
model_type=args.model_type
images_path=os.path.join(args.input,"*.*")
result_path=args.output
pretrain_dir=args.model_path
output_ch=args.output_ch



# def tensor_to_heatmap(feature_tensor, save_path):
#     """
#     将特征张量转换为热力图并保存
#     :param feature_tensor: 特征张量 (C, H, W)
#     :param save_path: 保存路径
#     """
#     # 转换为numpy数组
#     feature_map = feature_tensor.detach().cpu().numpy()
    
#     # 平均多通道（如果通道数>1）
#     if feature_map.shape[0] > 1:
#         heatmap = np.mean(feature_map, axis=0)
#     else:
#         heatmap = feature_map[0]
    
#     # 归一化
#     heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    
#     # 创建图像
#     plt.figure(figsize=(10, 10))
#     plt.imshow(heatmap, cmap='jet')
#     plt.axis('off')
    
#     # 保存图像
#     plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
#     plt.close()


# def forward_hook(module, input, output):
#     """捕获特征图的钩子函数"""
#     global feature_maps
#     feature_maps.append(output)


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
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)

def load_params(model_path):
    #  full_model=torch.load(model_path)
     full_model=torch.load(model_path, map_location=torch.device('cpu'))
     if 'params_ema' in full_model:
          return full_model['params_ema']
     elif 'params' in full_model:
          return full_model['params']
     else:
          return full_model

class ImageProcessor:
    def __init__(self, model):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Decom_l = Decom().cuda()
        self.Decom_l = aux_load_initialize(self.Decom_l,'~/pertained')
        self.Decom_l.eval()

    def resize_image(self, image, target_size):
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if original_width < original_height:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)

        return image.resize((new_width, new_height))

    def process_image(self, image):
        # Open the original image
        to_tensor=transforms.ToTensor()
        original_image = image

        # Resize the image proportionally to make the shorter side 512 pixels
        resized_image = self.resize_image(original_image, 512)
        resized_width, resized_height = resized_image.size

        # Process each 512-pixel segment separately
        segments = []
        overlaps = []
        if resized_width > 512:
            for end_x in range(512, resized_width+256, 256):
                end_x = min(end_x, resized_width)
                overlaps.append(end_x)
                cropped_image = resized_image.crop((end_x-512, 0, end_x, 512))
                cropped_image=to_tensor(cropped_image).unsqueeze(0).to(self.device)
                retinex = self.Decom_l(cropped_image)
                processed_segment = self.model(cropped_image,retinex).squeeze(0)
                segments.append(processed_segment)
        else:
            for end_y in range(512, resized_height+256, 256):
                end_y = min(end_y, resized_height)
                overlaps.append(end_y)
                cropped_image = resized_image.crop((0, end_y-512, 512, end_y))
                cropped_image=to_tensor(cropped_image).unsqueeze(0).to(self.device)
                retinex = self.Decom_l(cropped_image)
                processed_segment = self.model(cropped_image,retinex).squeeze(0)

                segments.append(processed_segment)
        overlaps = [0] + [prev - cur + 512 for prev, cur in zip(overlaps[:-1], overlaps[1:])]

        # Blending the segments
        for i in range(1, len(segments)):
            overlap = overlaps[i]
            alpha = torch.linspace(0, 1, steps=overlap).to(self.device)
            if resized_width > 512:
                alpha = alpha.view(1, -1, 1).expand(512, -1, 6).permute(2,0,1)
                segments[i][:, :, :overlap] = alpha * segments[i][:, :, :overlap] + (1 - alpha) * segments[i-1][:, :, -overlap:]
            else:
                alpha = alpha.view(-1, 1, 1).expand(-1, 512, 6).permute(2,0,1)
                segments[i][:, :overlap, :] = alpha * segments[i][:, :overlap, :] + (1 - alpha) * segments[i-1][:, -overlap:, :]

        # Concatenating all the segments
        if resized_width > 512:
            blended = [segment[:,:,:-overlap] for segment, overlap in zip(segments[:-1], overlaps[1:])] + [segments[-1]]
            merged_image = torch.cat(blended, dim=2)
        else:
            blended = [segment[:,:-overlap,:] for segment, overlap in zip(segments[:-1], overlaps[1:])] + [segments[-1]]
            merged_image = torch.cat(blended, dim=1)

        return merged_image

def demo(images_path,output_path,model_type,output_ch,pretrain_dir,flare7kpp_flag):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    test_path=sorted(glob.glob(images_path))
    result_path=output_path
    os.makedirs(result_path, exist_ok=True)
    torch.cuda.empty_cache()
    if model_type=='Uformer':
        model=Uformer(img_size=512,img_ch=3,output_ch=output_ch).cuda()
        model.load_state_dict(load_params(pretrain_dir))
        input_tensor = torch.randn(1, 3, 512, 512).cuda()
        re_tensor = torch.randn(1, 4, 512, 512).cuda()
        flops, params = profile(model, inputs=(input_tensor,re_tensor))
        print(f'FLOPs: {flops}, Params: {params}')

        #stat(model, (3,224,224))
    elif model_type=='U_Net' or model_type=='U-Net':
        model=U_Net(img_ch=3,output_ch=output_ch).cuda()
        model.load_state_dict(load_params(pretrain_dir))
    elif model_type=='SpA_former' or model_type=='SpA_former':
        model=SpA_former(output_ch=output_ch).cuda()
        model.load_state_dict(load_params(pretrain_dir))
    else:
        assert False, "This model is not supported!!"
    processor=ImageProcessor(model)
    to_tensor=transforms.ToTensor()

    for i,image_path in tqdm(enumerate(test_path)):
        # global feature_maps
        # feature_maps = []
        img_name = os.path.basename(image_path)
        if not flare7kpp_flag:
            mkdir(os.path.join(result_path,"deflare/"))
            deflare_path = os.path.join(result_path,"deflare/",img_name)

        mkdir(os.path.join(result_path,"flare/"))
        mkdir(os.path.join(result_path,"blend/"))
        mkdir(os.path.join(result_path,"cmp/"))
        
        flare_path = os.path.join(result_path,"flare/",img_name)
        blend_path = os.path.join(result_path,"blend/",img_name)

        merge_img = Image.open(image_path).convert("RGB")

        model.eval()
        #model.decoderlayer_3.blocks[0].register_forward_hook(forward_hook)
  

        with torch.no_grad():
            output_img=processor.process_image(merge_img).unsqueeze(0)
            #transform = transforms.ToTensor()
            #image_tensor = transform(output_img)
            gamma=torch.Tensor([2.2])
            if output_ch==6:
                deflare_img,flare_img_predicted,merge_img_predicted=predict_flare_from_6_channel(output_img,gamma)
            elif output_ch==3:
                flare_mask=torch.zeros_like(output_img)
                deflare_img,flare_img_predicted=predict_flare_from_3_channel(output_img,flare_mask,output_img,merge_img,merge_img,gamma)
            else:
                assert False, "This output_ch is not supported!!"



            if not flare7kpp_flag:
                torchvision.utils.save_image(deflare_img, deflare_path)
                deflare_img= blend_light_source(to_tensor(processor.resize_image(merge_img,512)).cuda().unsqueeze(0), deflare_img, 0.95)                
            deflare_img_np=deflare_img.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            deflare_img_pil=Image.fromarray((deflare_img_np * 255).astype(np.uint8))
            flare_img_orig=ImageChops.difference(merge_img.resize(deflare_img_pil.size),deflare_img_pil)
            deflare_img_orig=ImageChops.difference(merge_img,flare_img_orig.resize(merge_img.size,resample=Image.BICUBIC))
            flare_img_orig.save(flare_path)
            deflare_img_orig.save(blend_path)

         
demo(images_path,result_path,model_type,output_ch,pretrain_dir,args.flare7kpp)
