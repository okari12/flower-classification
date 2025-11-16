import numpy as np
import os
from skimage import io, color, feature
from skimage.transform import resize
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed, dump
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. 配置和路径 ---
DATA_ROOT = './'
IMAGE_DIR = './jpg'
AUGMENTED_DIR = './jpg_augmented'  # 新增的增强数据路径
RANDOM_STATE = 42
N_JOBS = -1


# --- 2. 特征提取函数 ---
def extract_features(image_path, bins=(8, 8, 8), radius=3, n_points=24):
    """提取颜色直方图（HSV）和LBP纹理特征。"""
    try:
        image = io.imread(image_path)
        image_gray = color.rgb2gray(image)
        image_resized = resize(image_gray, (128, 128))

        # 特征 A: 颜色特征 (HSV 直方图)
        image_hsv = color.rgb2hsv(image)
        hist, _ = np.histogramdd(
            image_hsv.reshape(-1, 3), bins=bins, range=[(0, 1), (0, 1), (0, 1)], density=True
        )
        color_features = hist.flatten()

        # 特征 B: 纹理特征 (LBP)
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
        # print(f"Error extracting features from {image_path}: {e}")
        return None


# --- 3. 数据加载和索引处理 ---

def get_image_paths_and_labels(data_root, image_dir, augmented_dir):
    """根据官方索引文件加载图片路径和标签，并划分数据集。"""
    label_struct = loadmat(os.path.join(data_root, 'imagelabels.mat'))
    all_labels = label_struct['labels'][0] - 1  # 转换为 0-based

    setid_struct = loadmat(os.path.join(data_root, 'setid.mat'))
    train_indices = setid_struct['trnid'][0] - 1
    val_indices = setid_struct['valid'][0] - 1
    test_indices = setid_struct['tstid'][0] - 1

    train_val_indices = np.concatenate([train_indices, val_indices])

    # --- 训练集 (使用增强后的图片) ---
    X_train_paths = []
    y_train_list = []

    for index_0based in train_val_indices:
        file_num = index_0based + 1
        label = all_labels[index_0based]

        # 查找所有增强后的图片 (e.g., image_00001_aug0.jpg, image_00001_aug1.jpg)
        for filename in os.listdir(augmented_dir):
            if filename.startswith(f'image_{file_num:05d}_aug'):
                X_train_paths.append(os.path.join(augmented_dir, filename))
                y_train_list.append(label)

    y_train = np.array(y_train_list)

    # --- 测试集 (使用原始图片) ---
    def get_path_original(index_0based):
        file_num = index_0based + 1
        return os.path.join(image_dir, f'image_{file_num:05d}.jpg')

    X_test_paths = [get_path_original(i) for i in test_indices]
    y_test = all_labels[test_indices]

    print(f"增强后的训练集/验证集样本总数: {len(X_train_paths)}")
    print(f"原始测试集样本总数: {len(X_test_paths)}")
    return X_train_paths, y_train, X_test_paths, y_test


# --- 4. 执行数据加载和并行特征提取 ---

print("开始加载数据并构建数据集划分...")
X_train_paths, y_train, X_test_paths, y_test = get_image_paths_and_labels(DATA_ROOT, IMAGE_DIR, AUGMENTED_DIR)

print("\n开始并行提取增强后的训练集特征 (耗时较长)...")
X_train_features = Parallel(n_jobs=N_JOBS)(
    delayed(extract_features)(path) for path in X_train_paths
)
valid_train_indices = [i for i, f in enumerate(X_train_features) if f is not None]
X_train = np.array([X_train_features[i] for i in valid_train_indices])
y_train = y_train[valid_train_indices]

print("开始并行提取测试集特征...")
X_test_features = Parallel(n_jobs=N_JOBS)(
    delayed(extract_features)(path) for path in X_test_paths
)
valid_test_indices = [i for i, f in enumerate(X_test_features) if f is not None]
X_test = np.array([X_test_features[i] for i in valid_test_indices])
y_test = y_test[valid_test_indices]

print(f"\n最终训练样本数: {X_train.shape[0]}, 测试样本数: {X_test.shape[0]}")
print(f"特征维度: {X_train.shape[1]}")

# --- 5. 数据预处理 (标准化) ---
print("进行特征标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 6. 随机森林模型调优 (Grid Search - 方法 3) ---

# 定义更广的参数网格
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [10, 20, None],
    'max_features': ['sqrt', 0.1, 0.2],
    'min_samples_leaf': [1, 5]
}
# --- Grid Search 结果 ---
# 最佳参数组合: {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 500}
# 最佳交叉验证准确率 (Top-1): 53.21%
# 最佳模型训练完成！
#
# 模型和标准化器已保存到 best_flower_model_v2.joblib 和 scaler_v2.joblib
#
# --- 最终优化模型评估结果 ---
# 测试集 Top-5 准确率 (Accuracy): 70.37%

rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)

print("\n开始 Grid Search 超参数调优 (耗时较长)...")
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=N_JOBS
)
grid_search.fit(X_train_scaled, y_train)

print("\n--- Grid Search 结果 ---")
print(f"最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证准确率 (Top-1): {grid_search.best_score_ * 100:.2f}%")

best_rf_model = grid_search.best_estimator_
print("最佳模型训练完成！")

# --- 7. 模型保存 ---
MODEL_PATH = 'best_flower_model_v2.joblib'
SCALER_PATH = 'scaler_v2.joblib'
dump(best_rf_model, MODEL_PATH)
dump(scaler, SCALER_PATH)
print(f"\n模型和标准化器已保存到 {MODEL_PATH} 和 {SCALER_PATH}")

# --- 8. 模型评估（仅 Top-5 Accuracy） ---

# 1. 预测所有测试样本的概率
y_proba = best_rf_model.predict_proba(X_test_scaled)

# 2. 找到每个样本的前 5 个最高概率的索引（类别）
top_5_indices = np.argsort(y_proba, axis=1)[:, ::-1][:, :5]

# 3. 检查每个样本的真实标签是否在前 5 个预测中
correct_predictions_top5 = 0
total_samples = len(y_test)

for i in range(total_samples):
    if y_test[i] in top_5_indices[i]:
        correct_predictions_top5 += 1

# 4. 计算 Top-5 准确率
accuracy_top5 = correct_predictions_top5 / total_samples

print(f"\n--- 最终优化模型评估结果 ---")
print(f"测试集 Top-5 准确率 (Accuracy): {accuracy_top5 * 100:.2f}%")