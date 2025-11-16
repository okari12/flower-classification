import numpy as np
import os
from skimage import io, color, feature
from skimage.transform import resize
from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV  # 导入网格搜索工具
from sklearn.metrics import classification_report, accuracy_score # 重新导入 classification_report
from joblib import Parallel, delayed, dump
import warnings

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. 配置和路径 ---
DATA_ROOT = './'
IMAGE_DIR = './jpg'
RANDOM_STATE = 42
N_JOBS = -1  # 使用所有核心进行并行计算

# --- 2. 特征提取函数（Color + LBP，保持最优配置） ---

def extract_features(image_path, bins=(8, 8, 8), radius=3, n_points=24):
    """
    提取颜色直方图（HSV）和LBP纹理特征。
    """
    try:
        # 1. 读取图像
        image = io.imread(image_path)
        # 转换为灰度图和标准化尺寸
        image_gray = color.rgb2gray(image)
        image_resized = resize(image_gray, (128, 128))

        # --- 特征 A: 颜色特征 (HSV 直方图) ---
        image_hsv = color.rgb2hsv(image)
        hist, _ = np.histogramdd(
            image_hsv.reshape(-1, 3),
            bins=bins, range=[(0, 1), (0, 1), (0, 1)], density=True
        )
        color_features = hist.flatten()

        # --- 特征 B: 纹理特征 (LBP) ---
        lbp = feature.local_binary_pattern(
            (image_resized * 255).astype(np.uint8), n_points, radius, method="uniform"
        )
        (hist_lbp, _) = np.histogram(
            lbp.ravel(), bins=np.arange(0, n_points + 2), range=(0, n_points + 1)
        )
        hist_lbp = hist_lbp.astype("float")
        hist_lbp /= (hist_lbp.sum() + 1e-7)
        texture_features = hist_lbp

        # 4. 组合所有特征
        return np.hstack([color_features, texture_features])
    except Exception as e:
        print(f"Error extracting features from {image_path}: {e}")
        return None


# --- 3. 数据加载和索引处理 ---

def get_image_paths_and_labels(data_root, image_dir):
    """根据官方索引文件加载图片路径和标签，并划分数据集。"""
    label_struct = loadmat(os.path.join(data_root, 'imagelabels.mat'))
    all_labels = label_struct['labels'][0] - 1  # 转换为 0-based

    setid_struct = loadmat(os.path.join(data_root, 'setid.mat'))
    # 转换为 0-based
    train_indices = setid_struct['trnid'][0] - 1
    val_indices = setid_struct['valid'][0] - 1
    test_indices = setid_struct['tstid'][0] - 1
    train_val_indices = np.concatenate([train_indices, val_indices])

    def get_path(index_0based):
        file_num = index_0based + 1
        return os.path.join(image_dir, f'image_{file_num:05d}.jpg')

    X_train_paths = [get_path(i) for i in train_val_indices]
    y_train = all_labels[train_val_indices]
    X_test_paths = [get_path(i) for i in test_indices]
    y_test = all_labels[test_indices]

    print(f"训练集/验证集样本总数: {len(X_train_paths)}")
    print(f"测试集样本总数: {len(X_test_paths)}")
    return X_train_paths, y_train, X_test_paths, y_test


# --- 4. 执行数据加载和并行特征提取 ---

print("开始加载数据并构建数据集划分...")
X_train_paths, y_train, X_test_paths, y_test = get_image_paths_and_labels(DATA_ROOT, IMAGE_DIR)

print("\n开始并行提取训练集最优特征...")
X_train_features = Parallel(n_jobs=N_JOBS)(
    delayed(extract_features)(path) for path in X_train_paths
)
# 过滤掉提取失败的样本
valid_train_indices = [i for i, f in enumerate(X_train_features) if f is not None]
X_train = np.array([X_train_features[i] for i in valid_train_indices])
y_train = y_train[valid_train_indices]

print("开始并行提取测试集特征...")
X_test_features = Parallel(n_jobs=N_JOBS)(
    delayed(extract_features)(path) for path in X_test_paths
)
# 过滤掉提取失败的样本
valid_test_indices = [i for i, f in enumerate(X_test_features) if f is not None]
X_test = np.array([X_test_features[i] for i in valid_test_indices])
y_test = y_test[valid_test_indices]

print(f"\n初始训练集特征维度: {X_train.shape[1]}")

# --- 5. 数据预处理 (仅标准化) ---

print("进行特征标准化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- 6. 随机森林模型调优 (Grid Search) ---

# 定义要搜索的参数网格
param_grid = {
    'n_estimators': [100, 300],
    'max_depth': [10, None],
    'max_features': ['sqrt', 0.2]
}

# 初始化随机森林分类器
rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)

# 初始化 Grid Search
print("\n开始 Grid Search 超参数调优 (可能耗时较长)...")
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=2,
    n_jobs=N_JOBS
)

# 在训练集上进行调优
grid_search.fit(X_train_scaled, y_train)

# 打印最佳参数
print("\n--- Grid Search 结果 ---")
print(f"最佳参数组合: {grid_search.best_params_}")
print(f"最佳交叉验证准确率 (Top-1): {grid_search.best_score_ * 100:.2f}%")

# 使用最佳模型进行最终预测
best_rf_model = grid_search.best_estimator_
print("最佳模型训练完成！")

# --- 7. 模型保存 ---
MODEL_PATH = 'best_flower_model.joblib'
SCALER_PATH = 'scaler.joblib'
dump(best_rf_model, MODEL_PATH)
dump(scaler, SCALER_PATH)
print(f"\n模型和标准化器已保存到 {MODEL_PATH} 和 {SCALER_PATH}")


# --- 8. 模型评估（Top-5 Accuracy 和 Top-1 Metrics） ---

# --- A. 计算 Top-5 准确率 ---
y_proba = best_rf_model.predict_proba(X_test_scaled)
top_5_indices = np.argsort(y_proba, axis=1)[:, ::-1][:, :5]

correct_predictions_top5 = 0
total_samples = len(y_test)

for i in range(total_samples):
    if y_test[i] in top_5_indices[i]:
        correct_predictions_top5 += 1

accuracy_top5 = correct_predictions_top5 / total_samples

# --- B. 计算 Top-1 Metrics ---
# 必须基于 Top-1 预测来计算精确率、召回率和 F1-Score
y_pred_top1 = best_rf_model.predict(X_test_scaled)
accuracy_top1 = accuracy_score(y_test, y_pred_top1)

print(f"\n--- 最终优化模型评估结果 ---")
print(f"测试集 Top-1 准确率 (Accuracy): {accuracy_top1 * 100:.2f}%")
print(f"测试集 Top-5 准确率 (Accuracy): {accuracy_top5 * 100:.2f}%")


print("\n详细分类指标 (基于 Top-1 预测的 Weighted Avg):")
# 使用 classification_report 获取精确率、召回率和 F1-Score
report = classification_report(y_test, y_pred_top1, digits=4, output_dict=True)

# 输出 Weighted Avg（加权平均）结果，适用于类别不平衡的多分类任务
print(f"  精确率 (Precision): {report['weighted avg']['precision']:.4f}")
print(f"  召回率 (Recall): {report['weighted avg']['recall']:.4f}")
print(f"  F1-Score (F1-Score): {report['weighted avg']['f1-score']:.4f}")