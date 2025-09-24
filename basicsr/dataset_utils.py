import torch
import numpy as np
import torch.nn.functional as F
import cv2
import random
import numpy as np
import os
import pywt
from PIL import Image

def Swap_Fre(im1, im2, xfm, ifm, component):
    im1_l, im1_h = xfm(im1)
    im2_l, im2_h = xfm(im2)
    im2_h[2] = im1_h[2]
    im2 = ifm((im2_l, im2_h))
    return im1, im2



def apply_augment(im1, im2, xfm, ifm, augs, mix_p, component):
    idx = np.random.choice(len(augs), p=mix_p)
    aug = augs[idx]
    mask = None

    if aug == "none":
        im1_aug, im2_aug = im1.clone(), im2.clone()
    elif aug == "Swap_Fre":
        im1_aug, im2_aug = Swap_Fre(
            im1.clone(), im2.clone(), xfm, ifm, component)
    else:
        raise ValueError("{} is not invalid.".format(aug))

    return im1_aug, im2_aug, mask, aug
    