<<<<<<< HEAD
# 花卉识别 Web 应用

这是一个基于 Flask 的花卉识别 Web 应用，它集成了两种不同的机器学习模型——传统模型（随机森林）和深度学习模型（MobileNetV2）——用于对上传的花卉图片进行分类。用户可以在前端界面上灵活选择使用哪个模型，并可以为每个模型独立启用测试时增强（TTA）功能以提高预测精度。

## ✨ 主要功能

- **双模型预测**：同时使用随机森林和 CNN (MobileNetV2) 进行预测，方便对比两者效果。
- **灵活的模型选择**：用户可以在前端自由选择使用一个或两个模型进行分析。
- **测试时增强 (TTA)**：为两个模型都提供了独立的 TTA 选项。启用后，应用会对原始图片及其增强版本（如翻转、旋转）进行多次预测并平均结果，以获得更稳健的识别效果。
- **实时结果展示**：在前端动态展示 Top-5 的预测结果，包括类别名称和置信度，并用进度条进行可视化。
- **清晰的前后端分离**：项目采用 Flask 作为后端，提供 API 接口；使用原生 HTML/CSS/JS 作为前端，结构清晰。

## 🛠️ 技术栈

- **后端**: Python, Flask
- **机器学习**: Scikit-learn, PyTorch
- **前端**: HTML, CSS, JavaScript
- **核心库**: NumPy, Joblib, Scikit-image, PIL

## 📂 项目结构

```
.
├── app.py                  # Flask 后端主应用
├── train_model.py          # 传统模型（随机森林）的训练脚本
├── train_cnn.py            # CNN 模型的训练脚本
├── requirements.txt        # 项目依赖库
├── templates/
│   └── index.html          # 前端页面
├── jpg/                      # 原始图片数据集 (未上传至 Git)
├── best_flower_model_v2.joblib # 训练好的随机森林模型 (未上传)
├── scaler_v2.joblib        # 对应的标准化器 (未上传)
├── flower_cnn_model.pth    # 训练好的 CNN 模型 (未上传)
└── README.md               # 项目说明文档
```

## 🚀 快速开始

### 1. 环境准备

- **克隆仓库**
  ```bash
  git clone https://github.com/okari12/flower-classification.git
  cd flower-classification
  ```

- **创建并激活虚拟环境** (推荐)
  ```bash
  # Windows
  python -m venv venv
  venv\Scripts\activate

  # macOS / Linux
  python3 -m venv venv
  source venv/bin/activate
  ```

- **安装依赖**
  ```bash
  pip install -r requirements.txt
  ```

### 2. 准备数据和模型

**重要提示**：由于数据集和模型文件体积较大，它们并未上传到 GitHub 仓库。您需要手动将它们放置在项目的根目录下。

- **数据集**：请确保 `jpg/` 文件夹存在于项目根目录，并且包含所有花卉图片。
- **标签与数据划分文件**：请确保 `imagelabels.mat` 和 `setid.mat` 文件在项目根目录。
- **模型文件**：
  - 如果您已经有训练好的模型，请将它们放在项目根目录：
    - `best_flower_model_v2.joblib`
    - `scaler_v2.joblib`
    - `flower_cnn_model.pth`
  - 如果没有，请运行训练脚本生成它们（见下一步）。

### 3. 训练模型 (可选)

如果您需要重新训练模型，可以运行以下脚本：

- **训练传统模型 (随机森林)**
  ```bash
  python train_model.py
  ```
  这会生成 `best_flower_model_v2.joblib` 和 `scaler_v2.joblib`。

- **训练 CNN 模型**
  ```bash
  python train_cnn.py
  ```
  这会生成 `flower_cnn_model.pth`。请确保您的机器支持 PyTorch（如有 GPU，会自动使用 CUDA）。

### 4. 启动 Web 应用

确保所有模型和数据文件都已就位后，运行主应用：

```bash
python app.py
```

启动成功后，您会看到类似以下的输出：
```
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```
现在，在您的浏览器中打开 `http://127.0.0.1:5000/` 即可开始使用。

