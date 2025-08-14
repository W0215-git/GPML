

import numpy as np
from skimage.feature import graycomatrix, graycoprops
from function.feature_extractors import fetureDIF
from function.feature_extractors import hog_features_patches as hog_features
from function.feature_extractors import histLBP
import sift_features
import numpy
import pywt
from scipy.stats import kurtosis, skew, entropy

def root_con_number3(v1, v2, v3):
    return numpy.asarray([v1, v2, v3])

def root_con_number2(v1, v2):
    return numpy.asarray([v1, v2])

def root_con2(v1, v2):
    v11 = numpy.concatenate((v1))
    v12 = numpy.concatenate((v2))
    feature_vector=numpy.concatenate((v11, v12),axis=0)
    return feature_vector

def root_con3(v1, v2, v3):
    v11 = numpy.concatenate((v1))
    v12 = numpy.concatenate((v2))
    v13 = numpy.concatenate((v3))
    feature_vector=numpy.concatenate((v11, v12, v13),axis=0)
    return feature_vector

def root_con(*args):
    feature_vector=numpy.concatenate((args),axis=0)
    return feature_vector


def all_dif(image, channel_idx):
    image = select_channel(image, channel_idx)
    # global and local
    feature_vector = fetureDIF(image)
    # 归一化
    return normalize_features(feature_vector)

def all_histogram(image, channel_idx):
    image = select_channel(image, channel_idx)
    # global and local
    n_bins = 32
    hist, ax = np.histogram(image, n_bins, [0, 255])
    # 归一化
    return normalize_features(hist)

def global_hog(image, channel_idx):
    image = select_channel(image, channel_idx)
    feature_vector = hog_features(image, 20, 10)
    # 归一化
    return normalize_features(feature_vector)

def local_hog(image, channel_idx):
    image = select_channel(image, channel_idx)
    try:
        feature_vector = hog_features(image, 10, 10)
    except:
        feature_vector = np.concatenate(image)
    # 归一化
    return normalize_features(feature_vector)



def all_lbp(image, channel_idx):
    image = select_channel(image, channel_idx)
    # global and local
    feature_vector = histLBP(image, 1.5, 8)
    # 归一化
    return normalize_features(feature_vector)

def all_sift(image, channel_idx):
    image = select_channel(image, channel_idx)
    # global and local
    width, height = image.shape
    min_length = np.min((width, height))
    img = np.asarray(image[0:width, 0:height])
    extractor = sift_features.SingleSiftExtractor(min_length)
    feaArrSingle = extractor.process_image(img[0:min_length, 0:min_length])
    # dimension 128 for all images
    w, h = feaArrSingle.shape
    feature_vector = np.reshape(feaArrSingle, (h,))
    # 归一化
    return normalize_features(feature_vector)

# 统计特征计算函数
def calculate_statistics(coefficients):
    """ 计算8种统计特征：mean, std, entropy, kurtosis, energy, avg_energy, norm, skewness """
    coeff_flat = coefficients.flatten()
    stats = {
        "mean": np.mean(coeff_flat),
        "std": np.std(coeff_flat),
        "entropy": entropy(np.abs(coeff_flat) + 1e-10),  # 避免 log(0)
        "kurtosis": kurtosis(coeff_flat),
        "energy": np.sum(coeff_flat ** 2),
        "avg_energy": np.mean(coeff_flat ** 2),
        "norm": np.linalg.norm(coeff_flat),
        "skewness": skew(coeff_flat)
    }
    return list(stats.values())


# 小波特征提取函数 (单通道)
def gray_wavelet(image, channel_idx, wavelet='db1', level=3):
    image = select_channel(image, channel_idx)

    features = []

    # 进行小波分解
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # 提取各个小波系数的统计特征
    for coeff in coeffs:
        if isinstance(coeff, tuple):  # 高频系数 (cH, cV, cD)
            for subband in coeff:
                features.extend(calculate_statistics(subband))
        else:  # 低频系数 (cA)
            features.extend(calculate_statistics(coeff))

    return normalize_features(np.array(features))


def gray_glcm(image, channel_idx):
    image = select_channel(image, channel_idx)
    """
    提取单通道灰度图像的GLCM特征（对比度、能量、相关性、同质性），并返回拼接后的特征向量。

    参数:
        image (numpy.ndarray): 输入的灰度图像 (H x W)，数据类型为 uint8。

    返回:
        feature_vector (numpy.ndarray): 提取的GLCM特征向量。
    """

    # 计算 GLCM
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)

    # 提取 GLCM 统计特征
    contrast = graycoprops(glcm, 'contrast').flatten()
    energy = graycoprops(glcm, 'energy').flatten()
    correlation = graycoprops(glcm, 'correlation').flatten()
    homogeneity = graycoprops(glcm, 'homogeneity').flatten()

    # 拼接特征向量
    feature_vector = np.concatenate([contrast, energy, correlation, homogeneity])

    return normalize_features(feature_vector)



def regionS(left,x,y,windowSize):
    width, height = left.shape[0], left.shape[1]
    x_end = min(width, x+windowSize)
    y_end = min(height, y+windowSize)
    slice = left[x:x_end, y:y_end]
    return slice

def regionR(left, x, y, windowSize1,windowSize2):
    width, height = left.shape[0], left.shape[1]
    x_end = min(width, x + windowSize1)
    y_end = min(height, y + windowSize2)
    slice = left[x:x_end, y:y_end]
    return slice

def feature_length(ind, instances, toolbox):
    func=toolbox.compile(ind)
    try:
        feature_len = len(func(instances))
    except: feature_len=0
    return feature_len,


def select_channel(img, channel_idx):
    """
    从多通道图像中选择一个指定通道的数据。
    参数:
        img: 多通道图像 (256, 256, channels)。
        channel_idx: 通道索引 (0 <= channel_idx < channels)。
    返回:
        指定通道的数据，形状为 (256, 256)。
    """
    return img[:, :, channel_idx]


def normalize_features(features):
    """
    归一化特征值，将特征缩放到 [0, 1] 范围。
    参数:
        features: 输入的特征值数组（可以是列表或numpy数组）。
    返回:
        归一化后的特征值数组。
    """
    min_val = np.min(features)
    max_val = np.max(features)

    # 防止除以零的情况
    if max_val - min_val != 0:
        normalized_features = (features - min_val) / (max_val - min_val)
    else:
        normalized_features = features  # 如果特征值相同，返回原始特征

    return normalized_features
