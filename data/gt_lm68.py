import numpy as np
import os
from PIL import Image
from scipy.io import loadmat, savemat
import os.path as osp

try:
    from PIL.Image import Resampling
    RESAMPLING_METHOD = Resampling.BICUBIC
except ImportError:
    from PIL.Image import BICUBIC
    RESAMPLING_METHOD = BICUBIC

def load_lm3d():
    bfm_folder = '../BFM'
    Lm3D = loadmat(osp.join(bfm_folder, 'similarity_Lm3D_all.mat'))
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D

lm3d_std = load_lm3d()

def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)

    return t, s

def extract_5p(lm):
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    lm5p = np.stack([lm[lm_idx[0], :], np.mean(lm[lm_idx[[1, 2]], :], 0), np.mean(
        lm[lm_idx[[3, 4]], :], 0), lm[lm_idx[5], :], lm[lm_idx[6], :]], axis=0)
    lm5p = lm5p[[1, 2, 0, 3, 4], :]
    return lm5p

# resize and crop images for face reconstruction
def resize_n_crop_img(img, lm, t, s, target_size=224., mask=None):
    w0, h0 = img.size
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    img = img.resize((w, h), resample=RESAMPLING_METHOD)
    img = img.crop((left, up, right, below))

    if mask is not None:
        mask = mask.resize((w, h), resample=RESAMPLING_METHOD)
        mask = mask.crop((left, up, right, below))

    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                  t[1] + h0/2], axis=1)*s
    lm = lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    return img, lm, mask

def align_img(img, lm, lm3D, mask=None, target_size=224., rescale_factor=102.):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --PIL.Image  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        mask_new           --PIL.Image  (target_size, target_size)
    
    Parameters:
        img                --PIL.Image  (raw_H, raw_W, 3)
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (5, 3)
        mask               --PIL.Image  (raw_H, raw_W, 3)
    """

    w0, h0 = img.size
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    s = rescale_factor/s

    # processing the image
    img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    trans_params = np.array([w0, h0, s, t[0, 0], t[1, 0]])

    return trans_params, img_new, lm_new, mask_new


# 保存最终变换后的 landmarks
def process_and_save_landmarks(input_folder, output_folder, lm3d_std, img_height=224):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    lm_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    for lm_file in lm_files:
        lm_path = os.path.join(input_folder, lm_file)
        raw_lm = np.loadtxt(lm_path).astype(np.float32)

        img_name = lm_file.replace('.txt', '.jpg')
        img_path = os.path.join(os.path.dirname(input_folder), img_name)  
        
        # img_path = lm_path.replace('landmarks', 'images').replace('.txt', '.jpg')
        if not os.path.exists(img_path):
            print(f"Image for {lm_file} not found, skipping.")
            continue
        raw_img = Image.open(img_path).convert('RGB')
        _, _, transformed_lm, _ = align_img(raw_img, raw_lm, lm3d_std)
        transformed_lm[:, 1] = img_height - 1 - transformed_lm[:, 1]  
        output_lm_path = os.path.join(output_folder, lm_file)
        np.savetxt(output_lm_path, transformed_lm, fmt='%.4f')
        print(f"Transformed landmarks saved to {output_lm_path}")

# 示例调用
input_folder = '../../../dataset/deep3d/AFLW2000/img/landmarks'  
output_folder = '../../../dataset/deep3d/AFLW2000/img/gt_lm68'  


process_and_save_landmarks(input_folder, output_folder, lm3d_std)
