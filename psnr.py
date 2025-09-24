import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse

def gamma_correction(image, gamma=2.2):
    """
    对图像进行 gamma 校正：将线性空间转换到 sRGB 空间
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def get_valid_win_size(img, default_win_size=7):
    """
    根据图像尺寸计算合适的 win_size，确保 win_size 为奇数且不超过图像最小边长
    """
    h, w = img.shape[:2]
    min_dim = min(h, w)
    if min_dim < default_win_size:
        return min_dim if min_dim % 2 == 1 else max(1, min_dim - 1)
    return default_win_size

def compute_metrics(normal_img, gt_img, gamma=2.2):
    """
    计算 Normal 图像（未经过 gamma 校正）与 GroundTruth（已 gamma 校正）之间的 PSNR 与 SSIM。
    这里对 Normal 图像先做 gamma 校正，再与 GT 进行比较，以确保在相同颜色空间下计算指标。
    """
    # 对 Normal 图像应用 gamma 校正
    normal_img_gamma = gamma_correction(normal_img, gamma=gamma)
    
    # 计算 PSNR
    psnr_val = cv2.PSNR(gt_img, normal_img_gamma)
    
    # 根据图像尺寸自动确定 SSIM 的 win_size
    win_size = get_valid_win_size(gt_img, default_win_size=7)
    
    # 如果图像为彩色，使用新版接口指定 channel_axis=-1
    if len(gt_img.shape) == 3 and gt_img.shape[2] > 1:
        ssim_val, _ = ssim(gt_img, normal_img_gamma, full=True, win_size=win_size, channel_axis=-1)
    else:
        ssim_val, _ = ssim(gt_img, normal_img_gamma, full=True, win_size=win_size)
    
    return psnr_val, ssim_val

def evaluate_normal(gt_folder, normal_folder, gamma=2.2):
    """
    遍历 GroundTruth 与 Normal 文件夹中的图像，
    先对 Normal 图像做 gamma 校正，再计算每对图像的 PSNR 与 SSIM 指标，
    并输出平均结果。
    """
    exts = ('.png', '.jpg', '.bmp')
    gt_files = sorted([os.path.join(gt_folder, f) for f in os.listdir(gt_folder) if f.lower().endswith(exts)])
    normal_files = sorted([os.path.join(normal_folder, f) for f in os.listdir(normal_folder) if f.lower().endswith(exts)])
    
    if len(gt_files) != len(normal_files):
        print("错误：GroundTruth 图像和 Normal 图像数量不匹配！")
        return

    total_psnr = 0.0
    total_ssim = 0.0
    count = len(gt_files)
    
    for gt_path, normal_path in zip(gt_files, normal_files):
        gt_img = cv2.imread(gt_path)
        normal_img = cv2.imread(normal_path)
        
        # 若图像尺寸不一致，则将 Normal 图像调整为 GT 图像的尺寸
        if gt_img.shape != normal_img.shape:
            normal_img = cv2.resize(normal_img, (gt_img.shape[1], gt_img.shape[0]))
        
        try:
            psnr_val, ssim_val = compute_metrics(normal_img, gt_img, gamma=gamma)
        except Exception as e:
            print(f"处理 {gt_path} 与 {normal_path} 时出错：{e}")
            continue
        
        total_psnr += psnr_val
        total_ssim += ssim_val
        
        print(f"{os.path.basename(gt_path)} 对应 {os.path.basename(normal_path)} -> PSNR: {psnr_val:.2f}, SSIM: {ssim_val:.4f}")
    
    print("\n--- 评估结果 ---")
    print(f"平均 PSNR: {total_psnr / count:.4f}")
    print(f"平均 SSIM: {total_ssim / count:.4f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 Normal 图像（未经过 gamma 校正）与 GroundTruth 的 PSNR 与 SSIM")
    parser.add_argument("--gt_folder", type=str, default='/home/ubuntu/axproject/GSAD-main_2/dataset/LOLv1/eval15/high', help="参考图像文件夹路径")
    parser.add_argument("--normal_folder", type=str, default="/home/ubuntu/axproject/Flare7K_mm/result/lolv1/deflare1", help="待测图像文件夹路径")
    parser.add_argument("--gamma", type=float, default=1, help="用于 gamma 校正的 gamma 值，默认 2.2")
    args = parser.parse_args()
    
    evaluate_normal(args.gt_folder, args.normal_folder, gamma=args.gamma)

    




