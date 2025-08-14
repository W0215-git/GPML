from random import random

import numpy as np
import pywt
from function.feature_extractors import fetureDIF
from function.feature_extractors import hog_features_patches as hog_features
from function.feature_extractors import histLBP
import cv2
import numpy
from skimage.feature import hog
from scipy.stats import kurtosis, skew, entropy
from skimage.feature import graycomatrix, graycoprops

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

def preprocess_image(image):
    """检查图像维度，如果是 (H, W, 1)，则去掉最后一个维度。"""
    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]
    return image

def all_dif(image):
    image = preprocess_image(image)
    # global and local
    feature_vector = fetureDIF(image)
    # 归一化
    return normalize_features(feature_vector)

def all_histogram(image):
    image = preprocess_image(image)
    # global and local
    n_bins = 32
    hist, _ = np.histogram(image, n_bins, [0, 255])
    # 归一化
    return normalize_features(hist)

def global_hog(image):
    image = preprocess_image(image)
    feature_vector = hog_features(image, 20, 10)
    # 归一化
    return normalize_features(feature_vector)

def local_hog(image):
    image = preprocess_image(image)
    try:
        feature_vector = hog_features(image, 10, 10)
    except:
        feature_vector = np.concatenate(image)
    # 归一化
    return normalize_features(feature_vector)

def all_lbp(image):
    image = preprocess_image(image)
    # global and local
    feature_vector = histLBP(image, 1.5, 8)
    # 归一化
    return normalize_features(feature_vector)

# def all_sift(image):
    image = preprocess_image(image)
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

def all_sift(image):
    """
    提取单通道灰度图像的全局SIFT特征，并返回归一化后的特征向量。

    参数:
        image (numpy.ndarray): 输入的灰度图像 (H x W)，数据类型为 uint8。

    返回:
        feature_vector (numpy.ndarray): 归一化后的SIFT特征向量。
    """
    # 检查输入图像是否为单通道
    if len(image.shape) != 2:
        raise ValueError("输入图像必须是单通道的灰度图像 (H x W)。")

    # 初始化 SIFT
    sift = cv2.SIFT_create()

    # 计算 SIFT 特征
    keypoints, descriptors = sift.detectAndCompute(image, None)

    # 如果未检测到特征点，返回零向量
    if descriptors is None:
        return np.zeros(128)

    # 计算特征向量的均值（全局特征表示）
    feature_vector = np.mean(descriptors, axis=0)

    # # 归一化特征向量
    # feature_vector = feature_vector / np.linalg.norm(feature_vector)

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
def gray_wavelet(image, wavelet='db1', level=3):
    image = preprocess_image(image)

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


def gray_glcm(image):
    """
    提取单通道灰度图像的GLCM特征（对比度、能量、相关性、同质性），并返回拼接后的特征向量。

    参数:
        image (numpy.ndarray): 输入的灰度图像 (H x W)，数据类型为 uint8。

    返回:
        feature_vector (numpy.ndarray): 提取的GLCM特征向量。
    """
    image = preprocess_image(image)

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


if __name__ == "__main__":
    # 生成一个随机灰度图像 (100 x 100)
    image_gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    # 提取GLCM特征
    feature_vector = gray_wavelet(image_gray)

    # 打印特征向量
    print(f"Feature vector dimension: {feature_vector.shape[0]}")
    print("Feature vector:", feature_vector)