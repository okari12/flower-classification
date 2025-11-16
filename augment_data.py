import os
import numpy as np
from skimage import io, transform, util
from scipy.io import loadmat
import random
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- 配置 ---
DATA_ROOT = './'
IMAGE_DIR = './jpg'  # 原始图片目录
AUGMENTED_DIR = './jpg_augmented'  # 增强后图片保存目录
AUG_FACTOR = 4  # 每张原始图片生成的新样本数量 (1个原图 + 3个增强图)

if not os.path.exists(AUGMENTED_DIR):
    os.makedirs(AUGMENTED_DIR)


# --- 增强函数 ---
def augment_image(image):
    """应用随机的图像变换"""
    # 1. 随机水平翻转
    if random.choice([True, False]):
        image = np.fliplr(image)

    # 2. 随机旋转 (不超过 15 度)
    angle = random.uniform(-15, 15)
    image = transform.rotate(image, angle, resize=True, mode='edge')

    # 3. 随机缩放/裁剪
    # 随机裁剪 80%-90% 的区域，然后缩放回原尺寸
    rows, cols, _ = image.shape
    scale = random.uniform(0.8, 1.0)
    new_rows, new_cols = int(rows * scale), int(cols * scale)

    start_r = random.randint(0, rows - new_rows)
    start_c = random.randint(0, cols - new_cols)

    image = image[start_r:start_r + new_rows, start_c:start_c + new_cols]
    image = transform.resize(image, (rows, cols), anti_aliasing=True)

    # 4. 随机颜色抖动 (调整亮度)
    image_float = util.img_as_float(image)
    brightness_factor = random.uniform(0.8, 1.2)  # 0.8 到 1.2 之间随机
    image_float = np.clip(image_float * brightness_factor, 0, 1)

    return util.img_as_ubyte(image_float)  # 转换回 uint8 格式


# --- 数据处理 ---
def process_and_augment_set():
    """加载索引并对训练/验证集进行增强"""
    try:
        setid_struct = loadmat(os.path.join(DATA_ROOT, 'setid.mat'))
        # 提取训练集和验证集的 0-based 索引
        train_indices = setid_struct['trnid'][0] - 1
        val_indices = setid_struct['valid'][0] - 1
        train_val_indices = np.concatenate([train_indices, val_indices])
    except FileNotFoundError:
        print(f"错误: 未找到 setid.mat 文件，请检查路径 {DATA_ROOT}")
        return

    print(f"开始对 {len(train_val_indices)} 张图片进行增强，目标样本数: {len(train_val_indices) * AUG_FACTOR}。")

    for index_0based in train_val_indices:
        file_num = index_0based + 1
        original_path = os.path.join(IMAGE_DIR, f'image_{file_num:05d}.jpg')

        try:
            original_image = io.imread(original_path)
        except FileNotFoundError:
            continue

        # 生成 AUG_FACTOR 个增强样本 (第一个为原始图片)
        for i in range(AUG_FACTOR):
            if i == 0:
                # 保存原始图片
                augmented_image = original_image
            else:
                # 生成增强图片
                augmented_image = augment_image(original_image)

            # 保存，命名格式：image_00001_aug0.jpg, image_00001_aug1.jpg, ...
            save_path = os.path.join(AUGMENTED_DIR, f'image_{file_num:05d}_aug{i}.jpg')
            io.imsave(save_path, augmented_image)

    print("数据增强完成，请检查 jpg_augmented 文件夹。")


if __name__ == '__main__':
    process_and_augment_set()