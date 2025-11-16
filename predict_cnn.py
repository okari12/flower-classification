import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json

# --- 配置 ---
MODEL_PATH = 'flower_cnn_model.pth'
NUM_CLASSES = 102
INPUT_SIZE = 224
# 示例图片路径
TEST_IMAGE_PATH = './jpg/image_00001.jpg'

# 类别名称映射 (与 app.py 中一致)
# 注意：这里的索引是 0-based
CATEGORY_NAMES = {
    0: "粉色报春花", 1: "硬叶袋装兰", 2: "风铃草", 3: "香豌豆", 4: "英国万寿菊",
    5: "虎皮百合", 6: "月亮兰", 7: "天堂鸟", 8: "乌头", 9: "球形蓟",
    10: "金鱼草", 11: "款冬", 12: "帝王花", 13: "矛蓟", 14: "气球花 (桔梗)",
    15: "巨型白色马蹄莲", 16: "火百合", 17: "针垫花", 18: "贝母", 19: "红火炬",
    20: "复活节百合", 21: "无茎龙胆", 22: "朝鲜蓟", 23: "香石竹", 24: "银灌木",
    25: "圣诞玫瑰", 26: "巴贝通雏菊", 27: "雏菊", 28: "报春花", 29: "向日葵",
    30: "鸢尾花", 31: "银莲花", 32: "树罂粟", 33: "非洲勋章菊", 34: "杜鹃花",
    35: "睡莲", 36: "玫瑰", 37: "凌霄花", 38: "蒲公英", 39: "粉黄大丽花",
    40: "矮牵牛", 41: "野三色堇", 42: "主教大丽花", 43: "山桃草", 44: "天竺葵",
    45: "紫锥菊", 46: "秘鲁百合", 47: "大星草", 48: "暹罗郁金", 49: "拖鞋草",
    50: "百日草", 51: "白色马蹄莲", 52: "罂粟", 53: "橙眼非洲雏菊", 54: "大白花雏菊",
    55: "仙客来", 56: "墨西哥矮牵牛", 57: "凤梨科植物", 58: "毛茛", 59: "内华达州一枝黄花",
    60: "牵牛花", 61: "俄罗斯鼠尾草", 62: "好望角花", 63: "美人蕉", 64: "朱顶红",
    65: "桂花", 66: "福禄考", 67: "黑种草", 68: "万寿菊", 69: "红唇卡特兰",
    70: "仙人掌花", 71: "加州罂粟", 72: "蓝目菊", 73: "春番红花", 74: "耧斗菜",
    75: "沙漠玫瑰", 76: "墨西哥翠菊", 77: "胜利郁金香", 78: "黑心菊", 79: "香蜂草",
    80: "绣球花", 81: "非洲堇", 82: "耐寒天竺葵", 83: "金莲花", 84: "大天使百合",
    85: "天竺葵属", 86: "嘉兰百合", 87: "海滨一枝黄花", 88: "金鱼草 (重复)",
    89: "红掌", 90: "鸡蛋花", 91: "铁线莲", 92: "朱槿", 93: "三角梅",
    94: "山茶花", 95: "锦葵", 96: "墨西哥向日葵", 97: "凤梨科植物 (重复)",
    98: "天人菊", 99: "全叶印第安画笔", 100: "羽毛河布默", 101: "桂竹香",
}


# --- 1. 模型定义 (必须与训练时完全一致) ---
def get_model(num_classes):
    # 使用与训练时相同的模型结构
    model = models.mobilenet_v2(weights=None) # 不加载预训练权重，因为我们要加载自己的权重
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model

# --- 2. 图像预处理 ---
def preprocess_image(image_path):
    """加载并转换单张图片以用于预测"""
    transform = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    # 添加一个批次维度 (C, H, W) -> (B, C, H, W)
    image = image.unsqueeze(0)
    return image

# --- 3. 预测函数 ---
def predict(image_tensor, model, device, top_k=5):
    """对单个图像张量进行预测"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)

        # 使用 softmax 获取概率
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        # 获取 Top-K 概率和类别索引
        top_probs, top_indices = torch.topk(probabilities, top_k)

        # 转换为 numpy 数组
        top_probs = top_probs.cpu().numpy().flatten()
        top_indices = top_indices.cpu().numpy().flatten()

        results = []
        for i in range(top_k):
            class_name = CATEGORY_NAMES.get(top_indices[i], f"未知类别 {top_indices[i]}")
            confidence = top_probs[i] * 100
            results.append(f"{class_name}: {confidence:.2f}%")

        return results

# --- 4. 主函数 ---
if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件 '{MODEL_PATH}'。请先运行 train_cnn.py 进行训练。")
    else:
        # 设置设备
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # 加载模型
        cnn_model = get_model(NUM_CLASSES)
        cnn_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        cnn_model.to(device)
        print("CNN 模型加载成功。")

        # 预处理图片
        image_tensor = preprocess_image(TEST_IMAGE_PATH)
        print(f"\n正在预测图片: {TEST_IMAGE_PATH}")

        # 执行预测
        predictions = predict(image_tensor, cnn_model, device)

        # 打印结果
        print("\n--- Top 5 预测结果 ---")
        for p in predictions:
            print(p)

