import numpy as np
import os
from joblib import load
from skimage import io, color, feature
from skimage.transform import resize, rotate
import scipy.io as sio

# --- 1. 定义文件路径 (必须与保存时一致) ---
MODEL_PATH = 'best_flower_model.joblib'
SCALER_PATH = 'scaler.joblib'
# 💡 示例：要预测的新图片路径
NEW_IMAGE_PATH = './jpg/image_00001.jpg'  # 假设用您数据集中的第一张图片测试

# --- 2. 加载模型和标准化器 ---
try:
    best_rf_model = load(MODEL_PATH)
    scaler = load(SCALER_PATH)
    print("模型和标准化器加载成功。")
except FileNotFoundError:
    print(f"错误：未能找到模型或标准化器文件。请确保 {MODEL_PATH} 和 {SCALER_PATH} 存在。")
    exit()


# --- 3. 定义特征提取函数 (必须与训练时完全相同) ---

def extract_features(image_path, bins=(8, 8, 8), radius=3, n_points=24):
    """
    提取颜色直方图（HSV）和LBP纹理特征 (与训练时相同)。
    """
    try:
        image = io.imread(image_path)
        image_gray = color.rgb2gray(image)
        image_resized = resize(image_gray, (128, 128))

        # 颜色特征
        image_hsv = color.rgb2hsv(image)
        hist, _ = np.histogramdd(
            image_hsv.reshape(-1, 3), bins=bins, range=[(0, 1), (0, 1), (0, 1)], density=True
        )
        color_features = hist.flatten()

        # 纹理特征
        lbp = feature.local_binary_pattern(
            (image_resized * 255).astype(np.uint8), n_points, radius, method="uniform"
        )
        (hist_lbp, _) = np.histogram(
            lbp.ravel(), bins=np.arange(0, n_points + 2), range=(0, n_points + 1)
        )
        hist_lbp = hist_lbp.astype("float")
        hist_lbp /= (hist_lbp.sum() + 1e-7)
        texture_features = hist_lbp

        return np.hstack([color_features, texture_features])
    except Exception as e:
        print(f"图片特征提取失败: {e}")
        return None


# --- 4. 预测函数 ---

def predict_single_image(image_path, model, scaler):
    """
    对单张图片进行特征提取、标准化和预测。
    """
    if not os.path.exists(image_path):
        print(f"错误：图片文件未找到: {image_path}")
        return None, None

    # 1. 特征提取
    raw_features = extract_features(image_path)
    if raw_features is None:
        return None, None

    # 2. 转换成 2D 数组 (1 样本, N 特征)
    # Scikit-learn 模型要求输入必须是二维数组
    features_2d = raw_features.reshape(1, -1)

    # 3. 标准化 (使用训练时的 Scaler)
    scaled_features = scaler.transform(features_2d)

    # 4. 预测类别 (返回 0 到 101 的整数)
    prediction_index = model.predict(scaled_features)[0]

    # 5. 预测概率 (可选，用于置信度)
    probabilities = model.predict_proba(scaled_features)[0]
    confidence = np.max(probabilities)

    return prediction_index, confidence


def extract_features_from_array(image, bins=(8, 8, 8), radius=3, n_points=24):
    """跟训练时相同的特征提取逻辑，但接受 numpy ndarray 而非文件路径。"""
    try:
        if image.ndim == 2:
            image_gray = image
        else:
            image_gray = color.rgb2gray(image)
        image_resized = resize(image_gray, (128, 128))

        if image.ndim == 2:
            image_rgb = np.stack([image, image, image], axis=-1)
        else:
            image_rgb = image
        image_hsv = color.rgb2hsv(image_rgb)
        hist, _ = np.histogramdd(
            image_hsv.reshape(-1, 3), bins=bins, range=[(0, 1), (0, 1), (0, 1)], density=True
        )
        color_features = hist.flatten()

        lbp = feature.local_binary_pattern(
            (image_resized * 255).astype(np.uint8), n_points, radius, method="uniform"
        )
        (hist_lbp, _) = np.histogram(
            lbp.ravel(), bins=np.arange(0, n_points + 2), range=(0, n_points + 1)
        )
        hist_lbp = hist_lbp.astype("float")
        hist_lbp /= (hist_lbp.sum() + 1e-7)
        texture_features = hist_lbp

        return np.hstack([color_features, texture_features])
    except Exception as e:
        print(f"图片数组特征提取失败: {e}")
        return None


def _augment_images_from_array(img_arr):
    """返回增强后的 numpy 图像列表（包括原图）。
    一些增强：原始、水平翻转、旋转 ±15/±30、亮度缩放 0.9/1.1
    """
    augs = []
    try:
        # Ensure dtype is suitable for skimage functions
        arr = img_arr
        augs.append(arr)
        # 水平翻转
        try:
            augs.append(np.fliplr(arr))
        except Exception:
            pass
        # 旋转（preserve_range 保持数值范围）
        for angle in (15, -15):
            try:
                r = rotate(arr, angle, preserve_range=True).astype(arr.dtype)
                augs.append(r)
            except Exception:
                pass
    except Exception as e:
        print(f"生成增强图失败: {e}")
    return augs


def tta_average_probabilities(image_path, model, scaler, n_augs=None):
    """对单张图片执行 TTA 并返回平均概率向量。
    - image_path: 文件路径
    - model: 已加载分类器（需支持 predict_proba）
    - scaler: 标准化器
    - n_augs: 限制增强数量（None 则全部）
    返回: avg_probs (1D numpy array) 或 None
    """
    try:
        img = io.imread(image_path)
    except Exception as e:
        print(f"读取图片失败（TTA）: {e}")
        return None

    aug_imgs = _augment_images_from_array(img)
    if n_augs is not None:
        aug_imgs = aug_imgs[:n_augs]

    all_probs = []
    for arr in aug_imgs:
        feats = extract_features_from_array(arr)
        if feats is None:
            continue
        feats_2d = feats.reshape(1, -1)
        try:
            scaled = scaler.transform(feats_2d)
            probs = model.predict_proba(scaled)[0]
            all_probs.append(probs)
        except Exception as e:
            print(f"TTA 预测失败（单增强）: {e}")
            continue

    if not all_probs:
        return None

    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs


def get_probs_single(image_path, model, scaler):
    """对单张图片计算 predict_proba，返回概率向量或 None。"""
    if not os.path.exists(image_path):
        print(f"错误：图片文件未找到: {image_path}")
        return None
    feats = extract_features(image_path)
    if feats is None:
        return None
    feats_2d = feats.reshape(1, -1)
    try:
        scaled = scaler.transform(feats_2d)
        probs = model.predict_proba(scaled)[0]
        return probs
    except Exception as e:
        print(f"计算单张概率失败: {e}")
        return None


# --- 5. 执行预测 ---
USE_TTA = True
TOP_K = 5

print(f"\n开始预测图片: {NEW_IMAGE_PATH}")

if USE_TTA:
    print("使用 TTA (测试时增强) 模式进行预测...")
    probs = tta_average_probabilities(NEW_IMAGE_PATH, best_rf_model, scaler)
else:
    print("使用标准模式进行预测...")
    probs = get_probs_single(NEW_IMAGE_PATH, best_rf_model, scaler)

if probs is None:
    print("未能获得预测概率，预测失败。")
else:
    topk_idx = np.argsort(-probs)[:TOP_K]
    print(f"\n--- Top-{TOP_K} 候选 ---")
    for rank, idx in enumerate(topk_idx, start=1):
        prob = probs[idx]
        print(f"{rank}. 类别索引(0-based): {idx}, 实际类别(1-based): {idx + 1}, 概率: {prob * 100:.2f}%")

    top1_idx = topk_idx[0]
    print(f"\n--- 最终 Top-1 预测 ---")
    print(f"预测的类别索引 (0-based): {top1_idx}")
    print(f"对应的实际类别 (1-based): {top1_idx + 1}")
    print(f"预测置信度: {probs[top1_idx] * 100:.2f}%")
