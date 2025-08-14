import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
import cv2
import pywt
from scipy.stats import entropy, kurtosis, skew
from skimage.feature import hog
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

def root_con_number3(v1, v2, v3):
    return np.asarray([v1, v2, v3])

def root_con_number2(v1, v2):
    return np.asarray([v1, v2])

def root_con2(v1, v2):
    v11 = np.concatenate((v1))
    v12 = np.concatenate((v2))
    feature_vector=np.concatenate((v11, v12),axis=0)
    return feature_vector

def root_con3(v1, v2, v3):
    v11 = np.concatenate((v1))
    v12 = np.concatenate((v2))
    v13 = np.concatenate((v3))
    feature_vector=np.concatenate((v11, v12, v13),axis=0)
    return feature_vector

def root_con(*args):
    feature_vector=np.concatenate((args),axis=0)
    return feature_vector


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

def color_glcm(image):
    """
    提取RGB图像的GLCM特征（对比度、能量、相关性、同质性），并返回拼接后的特征向量。
    同时将RGB图像转换为HSV颜色空间，提取V通道并计算其GLCM特征。

    参数:
        image (numpy.ndarray): 输入的RGB图像（H x W x 3），数据类型为 uint8。

    返回:
        归一化feature_vector (numpy.ndarray): 拼接后的特征向量，包括RGB通道和HSV V通道的GLCM特征。64维度
    """
    # 检查输入图像是否为三通道
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入图像必须是三通道的RGB图像 (H x W x 3)。")

    # 分离RGB通道
    channels = [image[:, :, 0], image[:, :, 1], image[:, :, 2]]  # 分别提取R、G、B通道

    # 将RGB图像转换为HSV颜色空间，并提取V通道
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv_image[:, :, 2]  # 提取V通道

    # 初始化存储特征的字典
    features = {'contrast': [], 'energy': [], 'correlation': [], 'homogeneity': []}

    # 对每个RGB通道计算GLCM
    for channel in channels:
        glcm = graycomatrix(channel, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        features['contrast'].append(graycoprops(glcm, 'contrast'))
        features['energy'].append(graycoprops(glcm, 'energy'))
        features['correlation'].append(graycoprops(glcm, 'correlation'))
        features['homogeneity'].append(graycoprops(glcm, 'homogeneity'))

    # 对V通道计算GLCM
    glcm_v = graycomatrix(v_channel, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
    features['contrast'].append(graycoprops(glcm_v, 'contrast'))
    features['energy'].append(graycoprops(glcm_v, 'energy'))
    features['correlation'].append(graycoprops(glcm_v, 'correlation'))
    features['homogeneity'].append(graycoprops(glcm_v, 'homogeneity'))

    # 初始化一个空的特征向量列表
    feature_vector = []

    # 遍历每个通道的特征并展开为一维向量
    for channel_idx in range(4):  # 3个RGB通道 + 1个V通道
        for feature_name in ['contrast', 'energy', 'correlation', 'homogeneity']:
            feature_vector.extend(features[feature_name][channel_idx].flatten())  # 将矩阵转换为一维并添加到特征向量中

    # 转换为 NumPy 数组格式
    feature_vector = np.array(feature_vector)

    return normalize_features(feature_vector)


def color_lbp(image, radius=1, n_points=8, method='nri_uniform'):
    """
    提取RGB图像的LBP特征，并返回拼接后的特征向量。
    同时将RGB图像转换为HSV颜色空间，提取V通道并计算其LBP特征。

    参数:
        image (numpy.ndarray): 输入的RGB图像（H x W x 3），数据类型为 uint8。
        radius (int): LBP的半径。
        n_points (int): LBP的邻域点数。
        method (str): LBP的计算方法，支持 'default', 'ror', 'uniform', 'nri_uniform', 'var'。

    返回:
        feature_vector (numpy.ndarray): 拼接后的特征向量，包括RGB通道和V通道的LBP直方图。59*4=236维度
    """
    # 检查输入图像是否为三通道
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入图像必须是三通道的RGB图像 (H x W x 3)。")

    # 分离RGB通道
    channels = [image[:, :, 0], image[:, :, 1], image[:, :, 2]]  # 分别提取R、G、B通道

    # 将RGB图像转换为HSV颜色空间，并提取V通道
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv_image[:, :, 2]  # 提取V通道

    # 初始化存储LBP直方图的列表
    histograms = []

    # 对每个RGB通道计算LBP并生成直方图
    for channel in channels:
        lbp = local_binary_pattern(channel, n_points, radius, method)
        hist, _ = np.histogram(lbp, bins=59, range=(0, 59))  # 计算LBP直方图
        histograms.extend(hist)

    # 对V通道计算LBP并生成直方图
    lbp_v = local_binary_pattern(v_channel, n_points, radius, method)
    hist_v, _ = np.histogram(lbp_v, bins=59, range=(0, 59))  # 计算LBP直方图
    histograms.extend(hist_v)

    # 转换为 NumPy 数组格式
    feature_vector = np.array(histograms)

    return normalize_features(feature_vector)

def color_hist(image, n_bins=32):
    """
    提取RGB图像的直方图特征，并返回拼接后的特征向量。
    同时将RGB图像转换为HSV颜色空间，提取V通道并计算其直方图。

    参数:
        image (numpy.ndarray): 输入的RGB图像（H x W x 3），数据类型为 uint8。
        n_bins (int): 直方图的 bins 数量，默认为 32。

    返回:
        feature_vector (numpy.ndarray): 拼接后的特征向量，包括RGB通道和V通道的归一化直方图。
    """
    # 检查输入图像是否为三通道
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入图像必须是三通道的RGB图像 (H x W x 3)。")

    # 分离RGB通道
    channels = [image[:, :, 0], image[:, :, 1], image[:, :, 2]]  # 分别提取R、G、B通道

    # 将RGB图像转换为HSV颜色空间，并提取V通道
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv_image[:, :, 2]  # 提取V通道

    # 初始化存储直方图的列表
    histograms = []

    # 对每个RGB通道计算直方图并归一化
    for channel in channels:
        hist, _ = np.histogram(channel, bins=n_bins, range=(0, 256))  # 计算直方图
        hist = normalize_features(hist)  # 归一化
        histograms.extend(hist)

    # 对V通道计算直方图并归一化
    hist_v, _ = np.histogram(v_channel, bins=n_bins, range=(0, 256))  # 计算直方图
    hist_v = normalize_features(hist_v)  # 归一化
    histograms.extend(hist_v)

    # 转换为 NumPy 数组格式
    feature_vector = np.array(histograms)

    return feature_vector

def color_sift(image):
    """
    提取RGB图像的全局SIFT特征，并返回拼接后的特征向量。
    同时将RGB图像转换为HSV颜色空间，提取V通道并计算其全局SIFT特征。

    参数:
        image (numpy.ndarray): 输入的RGB图像（H x W x 3），数据类型为 uint8。

    返回:
        feature_vector (numpy.ndarray): 拼接后的特征向量，包括RGB通道和V通道的全局SIFT描述子。
    """
    # 检查输入图像是否为三通道
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入图像必须是三通道的RGB图像 (H x W x 3)。")

    # 分离RGB通道
    channels = [image[:, :, 0], image[:, :, 1], image[:, :, 2]]  # 分别提取R、G、B通道

    # 将RGB图像转换为HSV颜色空间，并提取V通道
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv_image[:, :, 2]  # 提取V通道

    # 初始化 SIFT
    sift = cv2.SIFT_create()

    # 初始化存储描述子的列表
    descriptors_list = []

    # 对每个RGB通道计算全局SIFT描述子
    for channel in channels:
        height, width = channel.shape
        patch_size = max(height, width)  # Patch 大小等于整张图像的最大边长
        keypoint = cv2.KeyPoint(x=width // 2, y=height // 2, size=patch_size)  # 图像中心为关键点
        _, descriptors = sift.compute(channel, [keypoint])  # 计算 SIFT 描述子
        descriptors_list.extend(descriptors.flatten())  # 将描述子展开为一维并添加到列表

    # 对V通道计算全局SIFT描述子
    height, width = v_channel.shape
    patch_size = max(height, width)  # Patch 大小等于整张图像的最大边长
    keypoint = cv2.KeyPoint(x=width // 2, y=height // 2, size=patch_size)  # 图像中心为关键点
    _, descriptors_v = sift.compute(v_channel, [keypoint])  # 计算 SIFT 描述子
    descriptors_list.extend(descriptors_v.flatten())  # 将描述子展开为一维并添加到列表

    # 转换为 NumPy 数组格式
    feature_vector = np.array(descriptors_list)

    return normalize_features(feature_vector)


def featureMeanStd(image):
    """
    计算图像的均值和标准差。

    参数:
        image (numpy.ndarray): 输入的灰度图像（H x W）。

    返回:
        mean (float): 图像的均值。
        std (float): 图像的标准差。
    """
    mean = np.mean(image)
    std = np.std(image)
    return mean, std


def featureDIF(image):
    feature=np.zeros((20))
    width,height=image.shape
    width1=int(width/2)
    height1=int(height/2)
    width2=int(width/4)
    height2=int(height/4)
    #A1B1C1D1
    feature[0],feature[1]=featureMeanStd(image)
    #A1E1OG1
    feature[2],feature[3]=featureMeanStd(image[0:width1,0:height1])
    #E1B1H1O
    feature[4],feature[5]=featureMeanStd(image[0:width1,height1:height])
    #G1OF1D1
    feature[6],feature[7]=featureMeanStd(image[width1:width,0:height1])
    #OH1C1F1
    feature[8],feature[9]=featureMeanStd(image[width1:width,height1:height])
    #A2B2C2D2
    feature[10],feature[11]=featureMeanStd(image[width2:(width2+width1),height2:(height1+height2)])
    #G1H1
    feature[12],feature[13]=featureMeanStd(image[width1,:])
    #E1F1
    feature[14],feature[15]=featureMeanStd(image[:,height1])
    #G2H2
    feature[16],feature[17]=featureMeanStd(image[width1,height2:(height1+height2)])
    #E2F2
    feature[18],feature[19]=featureMeanStd(image[width2:(width2+width1),height1])
    return feature


def color_dif(image):
    """
    提取RGB图像的DIF特征，并返回拼接后的特征向量。
    同时将RGB图像转换为HSV颜色空间，提取V通道并计算其DIF特征。

    参数:
        image (numpy.ndarray): 输入的RGB图像（H x W x 3），数据类型为 uint8。

    返回:
        feature_vector (numpy.ndarray): 拼接后的特征向量，包括RGB通道和V通道的DIF特征。
    """
    # 检查输入图像是否为三通道
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入图像必须是三通道的RGB图像 (H x W x 3)。")

    # 分离RGB通道
    channels = [image[:, :, 0], image[:, :, 1], image[:, :, 2]]  # 分别提取R、G、B通道

    # 将RGB图像转换为HSV颜色空间，并提取V通道
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv_image[:, :, 2]  # 提取V通道

    # 初始化存储DIF特征的列表
    dif_features = []

    # 对每个RGB通道计算DIF特征
    for channel in channels:
        dif = featureDIF(channel)  # 计算DIF特征
        dif_features.extend(dif)  # 将特征添加到列表

    # 对V通道计算DIF特征
    dif_v = featureDIF(v_channel)  # 计算DIF特征
    dif_features.extend(dif_v)  # 将特征添加到列表

    # 转换为 NumPy 数组格式
    feature_vector = np.array(dif_features)

    return normalize_features(feature_vector)



def calculate_statistics(coefficients):
    """
    计算8种统计特征：mean, std, entropy, kurtosis, energy, avg_energy, norm, skewness。

    参数:
        coefficients (numpy.ndarray): 输入的小波系数。

    返回:
        stats (list): 包含8种统计特征的列表。
    """
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

def color_wavelet(image, wavelet='db1', level=3):
    """
    对图像的每个颜色通道进行三层小波分解并计算统计特征。
    同时将RGB图像转换为HSV颜色空间，提取V通道并计算其小波特征。

    参数:
        image (numpy.ndarray): 输入的RGB图像（H x W x 3），数据类型为 uint8。
        wavelet (str): 使用的小波基函数，默认为 'db1'。
        level (int): 小波分解的层数，默认为 3。

    返回:
        feature_vector (numpy.ndarray): 拼接后的特征向量，包括RGB通道和V通道的小波统计特征。
    """
    # 检查输入图像是否为三通道
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("输入图像必须是三通道的RGB图像 (H x W x 3)。")

    # 分离RGB通道
    channels = [image[:, :, 0], image[:, :, 1], image[:, :, 2]]  # 分别提取R、G、B通道

    # 将RGB图像转换为HSV颜色空间，并提取V通道
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv_image[:, :, 2]  # 提取V通道

    # 初始化存储特征的列表
    features = []

    # 对每个RGB通道和V通道进行小波分解并计算统计特征
    for channel in channels + [v_channel]:
        coeffs = pywt.wavedec2(channel, wavelet, level=level)  # 小波分解
        for coeff in coeffs:
            if isinstance(coeff, tuple):  # 高频系数 (cH, cV, cD)
                for subband in coeff:
                    features.extend(calculate_statistics(subband))
            else:  # 低频系数 (cA)
                features.extend(calculate_statistics(coeff))

    # 转换为 NumPy 数组格式
    feature_vector = np.array(features)

    return normalize_features(feature_vector)


def HoGFeatures(image):
    img,realImage=hog(image,orientations=9, pixels_per_cell=(8, 8),
                cells_per_block=(3, 3), block_norm='L2-Hys', visualize=True,
                transform_sqrt=False, feature_vector=True)
    return realImage

def hog_features_patches(image,patch_size,moving_size):
    img=np.asarray(image)
    width, height = img.shape
    w = int(width / moving_size)
    h = int(height / moving_size)
    patch = []
    for i in range(0, w):
        for j in range(0, h):
            patch.append([moving_size * i, moving_size * j])
    hog_features = np.zeros((len(patch)))
    realImage=HoGFeatures(img)
    for i in range(len(patch)):
        hog_features[i] = np.mean(
            realImage[patch[i][0]:(patch[i][0] + patch_size), patch[i][1]:(patch[i][1] + patch_size)])
    return hog_features

def rgb_to_gray(image):
    """Convert a NumPy array RGB image to grayscale."""
    if image.ndim == 3 and image.shape[2] == 3:  # Check if the image has 3 channels
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:  # Already a single-channel image
        return image
    else:
        raise ValueError("Input image must be a 2D grayscale or 3-channel RGB image.")

def global_hog(image):
    """Global HOG feature extraction for a NumPy array image."""
    gray_image = rgb_to_gray(image)  # Convert to grayscale
    feature_vector = hog_features_patches(gray_image, 20, 10)  # Extract HOG features (assume this is your function)
    return normalize_features(feature_vector)  # Normalize the feature vector

def local_hog(image):
    """Local HOG feature extraction for a NumPy array image."""
    gray_image = rgb_to_gray(image)  # Convert to grayscale
    try:
        feature_vector = hog_features_patches(gray_image, 10, 10)  # Extract local HOG features
    except Exception as e:
        print(f"Error extracting local HOG features: {e}")
        feature_vector = np.concatenate(gray_image)  # Fallback if extraction fails
    return normalize_features(feature_vector)  # Normalize the feature vector

# 示例用法
if __name__ == "__main__":
    # 示例：创建一个随机的三通道图像 (H=100, W=100, 3)
    image = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)

    # 提取特征
    feature_vector = global_hog(image)

    # 打印拼接后的特征向量维度和内容
    print(f"Feature vector dimension: {feature_vector.shape[0]}")
    print("Feature vector:", feature_vector)