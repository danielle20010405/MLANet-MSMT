import os
import numpy as np
from scipy.io import loadmat
from math import sqrt
import pickle
import os.path as osp
from PIL import Image

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))

# 加载 yaw 列表
d = 'test.configs'
yaws_list = _load(osp.join(d, 'AFLW2000-3D.pose.npy'))

def load_pred_lm(mat_folder):
    """Load predicted landmarks from .mat files."""
    mat_files = [f for f in os.listdir(mat_folder) if f.endswith('.mat')]
    mat_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort by numeric value before the extension
    pred_lms = []

    for mat_file in mat_files:
        mat_path = os.path.join(mat_folder, mat_file)
        data = loadmat(mat_path)
        pred_lm = data['lm68']  # 关键点保存在 'lm68' 字段中
        pred_lms.append(pred_lm)

    return mat_files, pred_lms

def load_gt_lm(gt_folder):
    """Load ground truth landmarks from .txt files."""
    txt_files = [f for f in os.listdir(gt_folder) if f.endswith('.txt')]
    txt_files.sort(key=lambda x: int(x.split('.')[0]))  # Sort by numeric value before the extension
    gt_lms = []

    for txt_file in txt_files:
        txt_path = os.path.join(gt_folder, txt_file)
        gt_lm = np.loadtxt(txt_path).astype(np.float32)
        gt_lms.append(gt_lm)

    return txt_files, gt_lms

def filter_yaw_list_by_files(yaws_list, files):
    """Filter the yaw list based on the indices present in the given files."""
    indices = [int(f.split('.')[0]) for f in files]  # Extract indices from file names
    filtered_yaws = [yaws_list[i] for i in indices if i < len(yaws_list)]
    return filtered_yaws

def calc_nme_with_bbox_diag(pts68_pred, pts68_true):
    """Calculate NME normalized by bounding box diagonal length."""
    # 计算真实关键点的边界框
    minx, maxx = np.min(pts68_true[:, 0]), np.max(pts68_true[:, 0])
    miny, maxy = np.min(pts68_true[:, 1]), np.max(pts68_true[:, 1])
    llength = sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)  # 对角线长度

    dis = pts68_pred - pts68_true
    dis = np.sqrt(np.sum(np.power(dis, 2), axis=1))
    mean_dis = np.mean(dis)

    nme = mean_dis / llength
    return nme

def calc_nme_for_all_with_bbox_diag(pred_lms, gt_lms):
    """Calculate NME for all predicted and ground truth landmarks normalized by bounding box diagonal length."""
    nme_list = []
    for pred_lm, gt_lm in zip(pred_lms, gt_lms):
        nme = calc_nme_with_bbox_diag(pred_lm, gt_lm)
        nme_list.append(nme)
    return nme_list

def analyze_nme_with_yaw(nme_list, filtered_yaw_list):
    """Analyze NME based on yaw angles."""
    ind_yaw_1 = np.abs(filtered_yaw_list) <= 30
    ind_yaw_2 = np.bitwise_and(np.abs(filtered_yaw_list) > 30, np.abs(filtered_yaw_list) <= 60)
    ind_yaw_3 = np.abs(filtered_yaw_list) > 60

    nme_1 = np.array(nme_list)[ind_yaw_1]
    nme_2 = np.array(nme_list)[ind_yaw_2]
    nme_3 = np.array(nme_list)[ind_yaw_3]

    print("[0, 30] Mean NME:", np.mean(nme_1) * 100)
    print("[30, 60] Mean NME:", np.mean(nme_2) * 100)
    print("[60, 90] Mean NME:", np.mean(nme_3) * 100)

# 设置路径
pred_folder = '../../../checkpoints/deep3d/bottle_co-downaff/results/AFLW2000_mat'
gt_folder = '../../../dataset/deep3d/AFLW2000/img/gt_lm68'
img_folder = '../../../dataset/deep3d/AFLW2000/img'  # 图像目录

# 加载数据
pred_files, pred_lms = load_pred_lm(pred_folder)
gt_files, gt_lms = load_gt_lm(gt_folder)

# 打印文件数量和文件名示例，确保顺序匹配
print("Number of predicted files:", len(pred_files))
print("Number of ground truth files:", len(gt_files))
print("First 5 matched pairs:")
for pf, gf in zip(pred_files[:5], gt_files[:5]):
    print(f"Predicted: {pf}, Ground truth: {gf}")

# 检查文件数量匹配
if len(pred_files) != len(gt_files):
    print("Error: The number of predicted and ground truth files do not match.")
    exit()

# 根据文件名索引过滤 yaw 列表
filtered_yaws_list = filter_yaw_list_by_files(yaws_list, pred_files)

# 计算 NME
nme_list = calc_nme_for_all_with_bbox_diag(pred_lms, gt_lms)
print("Mean NME:", np.mean(nme_list) * 100)

# 基于 yaw 分析 NME
analyze_nme_with_yaw(nme_list, filtered_yaws_list)
