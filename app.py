import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
from skimage import io as skio, color, feature
from skimage.transform import resize, rotate
import warnings
from skimage.transform import rotate
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 忽略运行时警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 配置 ---
app = Flask(__name__)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- 模型路径 ---
RF_MODEL_PATH = 'best_flower_model_v2.joblib'
SCALER_PATH = 'scaler_v2.joblib'
CNN_MODEL_PATH = 'flower_cnn_model.pth'
INPUT_SIZE = 224  # CNN 输入尺寸
N_CLASSES = 102

CATEGORY_NAMES = {
    0: "粉色报春花",
    1: "硬叶袋装兰",
    2: "风铃草",
    3: "香豌豆",
    4: "英国万寿菊",
    5: "虎皮百合",
    6: "月亮兰",
    7: "天堂鸟",
    8: "乌头",
    9: "球形蓟",
    10: "金鱼草",
    11: "款冬",
    12: "帝王花",
    13: "矛蓟",
    14: "气球花 (桔梗)",
    15: "巨型白色马蹄莲",
    16: "火百合",
    17: "针垫花",
    18: "贝母 (或 皇冠贝母)",
    19: "红火炬",
    20: "复活节百合",
    21: "无茎龙胆",
    22: "朝鲜蓟",
    23: "香石竹 (Sweet William)",
    24: "银灌木",
    25: "圣诞玫瑰",
    26: "巴贝通雏菊",
    27: "雏菊",
    28: "报春花",
    29: "向日葵",
    30: "鸢尾花",
    31: "银莲花",
    32: "树罂粟",
    33: "非洲勋章菊",
    34: "杜鹃花",
    35: "睡莲",
    36: "玫瑰",
    37: "凌霄花",
    38: "蒲公英",
    39: "粉黄大丽花",
    40: "矮牵牛",
    41: "野三色堇",
    42: "主教大丽花",
    43: "山桃草",
    44: "天竺葵",
    45: "紫锥菊",
    46: "秘鲁百合",
    47: "大星草",
    48: "暹罗郁金",
    49: "拖鞋草",
    50: "百日草",
    51: "白色马蹄莲",
    52: "罂粟",
    53: "橙眼非洲雏菊",
    54: "大白花雏菊",
    55: "仙客来",
    56: "墨西哥矮牵牛",
    57: "凤梨科植物",
    58: "毛茛 (Buttercup)",
    59: "内华达州一枝黄花",
    60: "牵牛花",
    61: "俄罗斯鼠尾草",
    62: "好望角花",
    63: "美人蕉",
    64: "朱顶红",
    65: "桂花",
    66: "福禄考 (Garden Phlox)",
    67: "黑种草",
    68: "万寿菊",
    69: "红唇卡特兰",
    70: "仙人掌花",
    71: "加州罂粟",
    72: "蓝目菊",
    73: "春番红花",
    74: "耧斗菜",
    75: "沙漠玫瑰",
    76: "墨西哥翠菊",
    77: "胜利郁金香",
    78: "黑心菊",
    79: "香蜂草",
    80: "绣球花",
    81: "非洲堇",
    82: "耐寒天竺葵",
    83: "金莲花",
    84: "大天使百合",
    85: "天竺葵属 (Pelargonium)",
    86: "嘉兰百合",
    87: "海滨一枝黄花",
    88: "金鱼草 (重复类别)",
    89: "红掌",
    90: "鸡蛋花",
    91: "铁线莲",
    92: "朱槿 (扶桑)",
    93: "三角梅",
    94: "山茶花",
    95: "锦葵",
    96: "墨西哥向日葵",
    97: "凤梨科植物 (重复类别)",
    98: "天人菊",
    99: "全叶印第安画笔",
    100: "羽毛河布默 (地方品种)",
    101: "桂竹香",
}

# --- 2. 预测函数 ---

# --- 2.1 随机森林模型相关函数 ---
def extract_features_rf(image_data, bins=(8, 8, 8), radius=3, n_points=24):
    """(原 extract_features) 提取用于随机森林的特征。"""
    try:
        image = skio.imread(image_data)
        if image.ndim == 3 and image.shape[2] == 4:
            image = color.rgba2rgb(image)
        image_gray = color.rgb2gray(image)
        image_resized = resize(image_gray, (128, 128))
        image_hsv = color.rgb2hsv(image)
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
        app.logger.error(f"RF 特征提取失败: {e}")
        return None

def predict_rf(img_bytes, model, scaler, use_tta=False):
    """使用随机森林模型进行预测，可选 TTA。"""
    if not use_tta:
        # --- 标准预测 ---
        try:
            image_data = io.BytesIO(img_bytes)
            raw_features = extract_features_rf(image_data)
            if raw_features is None: return None
            features_2d = raw_features.reshape(1, -1)
            scaled_features = scaler.transform(features_2d)
            return model.predict_proba(scaled_features)[0]
        except Exception as e:
            app.logger.error(f"RF 标准预测失败: {e}")
            return None
    else:
        # --- TTA 预测 ---
        try:
            img_arr = skio.imread(io.BytesIO(img_bytes))
            aug_imgs = _augment_images_from_array(img_arr)

            all_probs = []
            for arr in aug_imgs:
                feats = extract_features_from_array(arr)
                if feats is None: continue
                feats_2d = feats.reshape(1, -1)
                scaled = scaler.transform(feats_2d)
                probs = model.predict_proba(scaled)[0]
                all_probs.append(probs)

            if not all_probs: return None
            return np.mean(all_probs, axis=0)
        except Exception as e:
            app.logger.error(f"RF TTA 预测失败: {e}")
            return None

def _augment_images_from_array(img_arr):
    """为 RF 模型生成增强图像数组列表。"""
    augs = []
    try:
        augs.append(img_arr)
        augs.append(np.fliplr(img_arr))
        for angle in (15, -15):
            augs.append(rotate(img_arr, angle, preserve_range=True).astype(img_arr.dtype))
    except Exception as e:
        app.logger.error(f"RF 图像增强失败: {e}")
    return augs

def extract_features_from_array(image, bins=(8, 8, 8), radius=3, n_points=24):
    """从 numpy 数组中提取 RF 特征。"""
    try:
        if image.ndim == 3 and image.shape[2] == 4:
            image = color.rgba2rgb(image)
        image_gray = color.rgb2gray(image)
        image_resized = resize(image_gray, (128, 128))
        image_hsv = color.rgb2hsv(image)
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
        app.logger.error(f"RF 数组特征提取失败: {e}")
        return None

# --- 2.2 CNN 模型相关函数 ---
def get_cnn_model(num_classes):
    """定义与训练时相同的 CNN 模型结构。"""
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

def preprocess_image_cnn(img_bytes):
    """为 CNN 准备图像张量。"""
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)

def predict_cnn(img_bytes, model, device, use_tta=False):
    """使用 CNN 模型进行预测，可选 TTA。"""
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # 定义基础和 TTA 变换
        base_transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        tta_transform = transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(p=1.0), # 水平翻转
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if not use_tta:
            # --- 标准预测 ---
            image_tensor = base_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
            return probabilities.cpu().numpy().flatten()
        else:
            # --- TTA 预测 ---
            tensors = [
                base_transform(image).unsqueeze(0),
                tta_transform(image).unsqueeze(0) # 增加翻转版本
            ]

            all_probs = []
            with torch.no_grad():
                for tensor in tensors:
                    outputs = model(tensor.to(device))
                    all_probs.append(torch.nn.functional.softmax(outputs, dim=1))

            # 平均所有 TTA 预测结果
            avg_probs = torch.mean(torch.cat(all_probs, dim=0), dim=0)
            return avg_probs.cpu().numpy().flatten()

    except Exception as e:
        app.logger.error(f"CNN 预测失败: {e}")
        return None

# --- 3. 模型加载 ---
RF_MODEL = None
SCALER = None
CNN_MODEL = None
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    RF_MODEL = load(RF_MODEL_PATH)
    SCALER = load(SCALER_PATH)
    print("随机森林模型加载成功。")
except FileNotFoundError:
    print(f"警告：未找到随机森林模型 {RF_MODEL_PATH} 或 {SCALER_PATH}")

try:
    CNN_MODEL = get_cnn_model(N_CLASSES)
    CNN_MODEL.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=DEVICE))
    CNN_MODEL.to(DEVICE)
    CNN_MODEL.eval()
    print(f"CNN 模型加载成功，使用设备: {DEVICE}")
except FileNotFoundError:
    print(f"警告：未找到 CNN 模型 {CNN_MODEL_PATH}。请先完成训练。")


# --- 4. 路由定义 ---

@app.route('/')
def index():
    """渲染前端页面"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """接收图片，根据选择使用一个或多个模型预测并返回结果"""
    if 'image' not in request.files:
        return jsonify({'error': '没有找到图片文件'}), 400
    file = request.files['image']
    if file.filename == '' or '.' not in file.filename or \
            file.filename.rsplit('.', 1)[1].lower() not in ALLOWED_EXTENSIONS:
        return jsonify({'error': '文件无效'}), 400

    try:
        # 从表单获取选项
        models_to_use = request.form.getlist('models[]') # 接收模型选择列表
        use_tta_rf = request.form.get('tta_rf', 'false').lower() == 'true'
        use_tta_cnn = request.form.get('tta_cnn', 'false').lower() == 'true'

        if not models_to_use:
            return jsonify({'error': '请至少选择一个模型'}), 400

        img_bytes = file.read()

        results_rf = []
        results_cnn = []

        # --- 模型 1: 随机森林预测 ---
        if 'rf' in models_to_use and RF_MODEL and SCALER:
            probs_rf = predict_rf(img_bytes, RF_MODEL, SCALER, use_tta=use_tta_rf)
            if probs_rf is not None:
                top_5_indices = np.argsort(probs_rf)[::-1][:5]
                for i in top_5_indices:
                    results_rf.append({
                        'name': CATEGORY_NAMES.get(i, f"类别 {i+1}"),
                        'confidence': f"{probs_rf[i] * 100:.2f}"
                    })

        # --- 模型 2: CNN 预测 ---
        if 'cnn' in models_to_use and CNN_MODEL:
            # 确保在读取字节后重置指针
            probs_cnn = predict_cnn(img_bytes, CNN_MODEL, DEVICE, use_tta=use_tta_cnn)
            if probs_cnn is not None:
                top_5_indices = np.argsort(probs_cnn)[::-1][:5]
                for i in top_5_indices:
                    results_cnn.append({
                        'name': CATEGORY_NAMES.get(i, f"类别 {i+1}"),
                        'confidence': f"{probs_cnn[i] * 100:.2f}"
                    })

        # 返回所选模型的结果
        return jsonify({
            'success': True,
            'results_rf': results_rf,
            'results_cnn': results_cnn
        })

    except Exception as e:
        app.logger.error(f"预测过程中发生错误: {e}")
        return jsonify({'error': f'服务器内部错误: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)