import random
import numpy as np
import gp_restrict
from strongGPDataType import Int1, Int2, Int3, Int4, Img, Region, Vector, Vector1, Vector2
import function.gray_feature_function as gff 
import function.color_feature_function as cff 
import function.multic_feature_function as mff 
from functools import partial
from deap import gp

def create_pset_gray(modality, bound1, bound2):
    pset_gray = gp.PrimitiveSetTyped(f'Tree_{modality}', [Img], Vector1, prefix=f'{modality}_img')
    pset_gray.addPrimitive(gff.root_con, [Vector1, Vector1], Vector1, name='FeaCon')
    pset_gray.addPrimitive(gff.root_con, [Vector, Vector], Vector1, name='FeaCon2')
    pset_gray.addPrimitive(gff.root_con, [Vector, Vector, Vector], Vector1, name='FeaCon3')
    pset_gray.addPrimitive(gff.all_dif, [Img], Vector, name='Global_DIF')
    pset_gray.addPrimitive(gff.all_histogram, [Img], Vector, name='Global_Histogram')
    pset_gray.addPrimitive(gff.global_hog, [Img], Vector, name='Global_HOG')
    pset_gray.addPrimitive(gff.all_lbp, [Img], Vector, name='Global_uLBP')
    pset_gray.addPrimitive(gff.all_sift, [Img], Vector, name='Global_SIFT')
    pset_gray.addPrimitive(gff.gray_glcm, [Img], Vector, name='Global_GLCM')
    pset_gray.addPrimitive(gff.gray_wavelet, [Img], Vector, name='Global_WaveLET')
    pset_gray.addPrimitive(gff.all_dif, [Region], Vector, name='Local_DIF')
    pset_gray.addPrimitive(gff.all_histogram, [Region], Vector, name='Local_Histogram')
    pset_gray.addPrimitive(gff.local_hog, [Region], Vector, name='Local_HOG')
    pset_gray.addPrimitive(gff.all_lbp, [Region], Vector, name='Local_uLBP')
    pset_gray.addPrimitive(gff.all_sift, [Region], Vector, name='Local_SIFT')
    pset_gray.addPrimitive(gff.gray_glcm, [Region], Vector, name='Global_GLCM')
    pset_gray.addPrimitive(gff.gray_wavelet, [Region], Vector, name='Global_WaveLET')
    pset_gray.addPrimitive(gff.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S')
    pset_gray.addPrimitive(gff.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R')
    pset_gray.renameArguments(ARG0=f'{modality}')
    pset_gray.addEphemeralConstant(f'X_{modality}', partial(random.randint, 0, bound1 - 20), Int1)
    pset_gray.addEphemeralConstant(f'Y_{modality}', partial(random.randint, 0, bound2 - 20), Int2)
    pset_gray.addEphemeralConstant(f'Size_{modality}', partial(random.randint, 20, 51), Int3)
    return pset_gray

def create_pset_color(modality, bound1, bound2):
    pset_color = gp.PrimitiveSetTyped(f'Tree_{modality}', [Img], Vector1, prefix=f'{modality}_img')
    pset_color.addPrimitive(cff.root_con, [Vector1, Vector1], Vector1, name='FeaCon')
    pset_color.addPrimitive(cff.root_con, [Vector, Vector], Vector1, name='FeaCon2')
    pset_color.addPrimitive(cff.root_con, [Vector, Vector, Vector], Vector1, name='FeaCon3')
    pset_color.addPrimitive(cff.color_dif, [Img], Vector, name='Global_CDIF')
    pset_color.addPrimitive(cff.color_hist, [Img], Vector, name='Global_CHist')
    pset_color.addPrimitive(cff.global_hog, [Img], Vector, name='Global_CHOG')
    pset_color.addPrimitive(cff.color_lbp, [Img], Vector, name='Global_CuLBP')
    pset_color.addPrimitive(cff.color_sift, [Img], Vector, name='Global_CSIFT')
    pset_color.addPrimitive(cff.color_glcm, [Img], Vector, name='Global_CGLCM')
    pset_color.addPrimitive(cff.color_wavelet, [Img], Vector, name='Global_CWaveLET')
    pset_color.addPrimitive(cff.color_dif, [Region], Vector, name='Local_CDIF')
    pset_color.addPrimitive(cff.color_hist, [Region], Vector, name='Local_CHist')
    pset_color.addPrimitive(cff.local_hog, [Region], Vector, name='Local_CHOG')
    pset_color.addPrimitive(cff.color_lbp, [Region], Vector, name='Local_CuLBP')
    pset_color.addPrimitive(cff.color_sift, [Region], Vector, name='Local_CSIFT')
    pset_color.addPrimitive(cff.color_glcm, [Region], Vector, name='Global_CGLCM')
    pset_color.addPrimitive(cff.color_wavelet, [Region], Vector, name='Global_CWaveLET')
    pset_color.addPrimitive(cff.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S')
    pset_color.addPrimitive(cff.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R')
    pset_color.renameArguments(ARG0=f'{modality}')
    pset_color.addEphemeralConstant(f'X_{modality}', partial(random.randint, 0, bound1 - 20), Int1)
    pset_color.addEphemeralConstant(f'Y_{modality}', partial(random.randint, 0, bound2 - 20), Int2)
    pset_color.addEphemeralConstant(f'Size_{modality}', partial(random.randint, 20, 51), Int3)
    return pset_color

def create_pset_multic(modality, bound1, bound2, channel_m2):
    pset = gp.PrimitiveSetTyped(f'Tree_{modality}', [Img], Vector1, prefix=f'{modality}_img')
    pset.addPrimitive(mff.root_con, [Vector1, Vector1], Vector1, name='FeaCon')
    pset.addPrimitive(mff.root_con, [Vector, Vector], Vector1, name='FeaCon2')
    pset.addPrimitive(mff.root_con, [Vector, Vector, Vector], Vector1, name='FeaCon3')
    pset.addPrimitive(mff.all_dif, [Img, Int4], Vector, name='Global_DIF')
    pset.addPrimitive(mff.all_histogram, [Img, Int4], Vector, name='Global_Histogram')
    pset.addPrimitive(mff.global_hog, [Img, Int4], Vector, name='Global_HOG')
    pset.addPrimitive(mff.all_lbp, [Img, Int4], Vector, name='Global_uLBP')
    pset.addPrimitive(mff.all_sift, [Img, Int4], Vector, name='Global_SIFT')
    pset.addPrimitive(mff.gray_glcm, [Img, Int4], Vector, name='Global_GLCM')
    pset.addPrimitive(mff.gray_wavelet, [Img, Int4], Vector, name='Global_WAVELET')
    pset.addPrimitive(mff.all_dif, [Region, Int4], Vector, name='Local_DIF')
    pset.addPrimitive(mff.all_histogram, [Region, Int4], Vector, name='Local_Histogram')
    pset.addPrimitive(mff.local_hog, [Region, Int4], Vector, name='Local_HOG')
    pset.addPrimitive(mff.all_lbp, [Region, Int4], Vector, name='Local_uLBP')
    pset.addPrimitive(mff.all_sift, [Region, Int4], Vector, name='Local_SIFT')
    pset.addPrimitive(mff.gray_glcm, [Region, Int4], Vector, name='Global_GLCM')
    pset.addPrimitive(mff.gray_wavelet, [Region, Int4], Vector, name='Global_WAVELET')
    pset.addPrimitive(mff.regionS, [Img, Int1, Int2, Int3], Region, name='Region_S')
    pset.addPrimitive(mff.regionR, [Img, Int1, Int2, Int3, Int3], Region, name='Region_R')
    pset.renameArguments(ARG0=f'{modality}')
    pset.addEphemeralConstant(f'X_{modality}', partial(random.randint, 0, bound1 - 20), Int1)
    pset.addEphemeralConstant(f'Y_{modality}', partial(random.randint, 0, bound2 - 20), Int2)
    pset.addEphemeralConstant(f'Size_{modality}', partial(random.randint, 20, 51), Int3)
    pset.addEphemeralConstant('ChannelIdx', lambda: random.randint(0, channel_m2 - 1), Int4)
    return pset

def pad_features(*vectors, mode='zero'):
    max_len = max(len(vec) for vec in vectors)
    padded_vectors = []
    for vec in vectors:
        if len(vec) < max_len:
            if mode == 'zero':
                vec = np.pad(vec, (0, max_len - len(vec)), mode='constant', constant_values=0)
            elif mode == 'repeat':
                vec = np.tile(vec, (max_len // len(vec) + 1))[:max_len]
        padded_vectors.append(vec)
    return tuple(padded_vectors)

def padded_add(*vectors):
    vectors = pad_features(*vectors)
    return np.sum(vectors, axis=0)

def padded_sub(*vectors):
    vectors = pad_features(*vectors)
    return vectors[0] - np.sum(vectors[1:], axis=0)

def padded_mul(*vectors):
    vectors = pad_features(*vectors)
    result = np.ones_like(vectors[0])
    for vec in vectors:
        result *= vec
    return result

def safe_div(x, y):
    return x / (y + 1e-8)

def padded_div(*vectors):
    vectors = pad_features(*vectors)
    result = vectors[0]
    for vec in vectors[1:]:
        result = np.vectorize(safe_div)(result, vec)
    return result

def padded_max(*vectors):
    vectors = pad_features(*vectors)
    return np.maximum.reduce(vectors)

def padded_avg(*vectors):
    vectors = pad_features(*vectors)
    return np.mean(vectors, axis=0)

def create_pset_fusion(num_modalities):
    input_types = [Vector] * num_modalities
    pset = gp.PrimitiveSetTyped("Tree", input_types, Vector, prefix="F")
    for i in range(2, num_modalities + 1):
        pset.addPrimitive(padded_add, [Vector] * i, Vector, name=f"Add_{i}")
        pset.addPrimitive(padded_mul, [Vector] * i, Vector, name=f"Mul_{i}")
        pset.addPrimitive(padded_max, [Vector] * i, Vector, name=f"Max_{i}")
        pset.addPrimitive(padded_avg, [Vector] * i, Vector, name=f"Avg_{i}")
    for i in range(num_modalities):
        pset.renameArguments(**{f"ARG{i}": f"Feature_Modality{i + 1}"})
    return pset

def initTrees(pset_list, pset_fusion, min_depth, max_depth):
    try:
        if not isinstance(pset_list, dict) or not pset_list:
            raise ValueError("pset_list must be a non-empty dictionary!")
        if not pset_fusion:
            raise ValueError("pset_fusion cannot be None or empty!")
        trees = {}
        for modality, pset in pset_list.items():
            tree = gp_restrict.genHalfAndHalfMD(pset=pset, min_=min_depth, max_=max_depth)
            if tree is None:
                raise ValueError(f"Failed to generate tree for modality: {modality}")
            if not isinstance(tree, gp.PrimitiveTree):
                tree = gp.PrimitiveTree(tree)
            trees[modality] = tree
        fusion_tree = gp_restrict.genHalfAndHalfMD(pset=pset_fusion, min_=2, max_=4)
        if fusion_tree is None:
            raise ValueError("Failed to generate fusion tree!")
        if not isinstance(fusion_tree, gp.PrimitiveTree):
            fusion_tree = gp.PrimitiveTree(fusion_tree)
        trees["fusion"] = fusion_tree
        return list(trees.values())
    except Exception as e:
        print(f"❌ Error in initTrees: {str(e)}")
        raise

def generate_random_tree(pset, max_depth=3):
    expr = gp.genFull(pset=pset, min_=1, max_=max_depth)
    tree = gp.PrimitiveTree(expr)
    return tree

if __name__ == '__main__':
    min_depth = 1
    max_depth = 3
    pset_list = {}
    modalities = ['flair', 't1', 't1ce', 't2']
    for modality in modalities:
        bound1, bound2, channel_count = 100, 100, 30
        if channel_count == 1:
            pset_list[modality] = create_pset_gray(modality, bound1, bound2, channel_count)
            print(f"✅ Created Gray-Scale GP tree for modality: {modality} (1 channel)")
        else:
            pset_list[modality] = create_pset_multic(modality, bound1, bound2, channel_count)
            print(f"✅ Created Multi-Channel GP tree for modality: {modality} ({channel_count} channels)")
    pset_fusion = create_pset_fusion(len(modalities))
    trees = initTrees(pset_list, pset_fusion, min_depth, max_depth)
    modality_names = list(pset_list.keys()) + ["fusion"]
    for i, tree in enumerate(trees):
        modality = modality_names[i]
        print(f" - {modality}: {tree}")
    fusion_tree = trees[-1]
    fusion_func = gp.compile(expr=fusion_tree, pset=pset_fusion)
    from strongGPDataType import Vector
    test_vectors = [
        np.array([1, 2, 3]),
        np.array([4, 5]),
        np.array([6, 7, 8, 9]),
        np.array([10])
    ]
    print("\n🌳 融合树结构:")
    print(fusion_tree)
    print("\n📥 融合输入向量:")
    for i, v in enumerate(test_vectors):
        print(f"Feature_Modality{i+1}: {v}")
    try:
        result = fusion_func(*test_vectors)
        print("\n📤 融合输出向量:")
        print(result)
    except Exception as e:
        print(f"❌ 融合树执行失败: {e}")
