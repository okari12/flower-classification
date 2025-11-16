import os
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

# --- 配置 ---
DATA_DIR = './'
IMAGE_DIR = os.path.join(DATA_DIR, 'jpg')
LABELS_PATH = os.path.join(DATA_DIR, 'imagelabels.mat')
SETID_PATH = os.path.join(DATA_DIR, 'setid.mat')

# 模型参数
MODEL_SAVE_PATH = 'flower_cnn_model.pth'
NUM_CLASSES = 102
BATCH_SIZE = 32
NUM_EPOCHS = 15  # 对于迁移学习，10-20 个 epoch 通常足够
LEARNING_RATE = 0.001
INPUT_SIZE = 224 # MobileNetV2 需要 224x224 的输入

# --- 1. 数据集定义 ---
class FlowerDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# --- 2. 数据预处理和加载 ---
def get_dataloaders():
    # 加载 MATLAB 文件
    labels_mat = scipy.io.loadmat(LABELS_PATH)['labels'][0]
    setid_mat = scipy.io.loadmat(SETID_PATH)

    # 标签是从 1 开始的，转换为从 0 开始
    labels = np.array(labels_mat) - 1

    # 获取训练、验证和测试集的文件索引 (1-based)
    train_indices = setid_mat['trnid'][0] - 1
    valid_indices = setid_mat['valid'][0] - 1

    # 生成完整的文件路径列表
    all_image_files = sorted([os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])

    train_paths = [all_image_files[i] for i in train_indices]
    valid_paths = [all_image_files[i] for i in valid_indices]

    train_labels = labels[train_indices]
    valid_labels = labels[valid_indices]

    # 定义数据增强和转换
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(INPUT_SIZE),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 创建数据集和数据加载器
    train_dataset = FlowerDataset(train_paths, train_labels, data_transforms['train'])
    valid_dataset = FlowerDataset(valid_paths, valid_labels, data_transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    return train_loader, valid_loader

# --- 3. 模型定义 ---
def get_model():
    # 加载预训练的 MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # 冻结所有基础层
    for param in model.parameters():
        param.requires_grad = False

    # 替换分类器
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)

    return model

# --- 4. 训练循环 ---
def train_model(model, train_loader, valid_loader, device):
    criterion = nn.CrossEntropyLoss()
    # 只优化我们新加的分类器层
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 验证阶段
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(valid_loader.dataset)
        epoch_acc = running_corrects.double() / len(valid_loader.dataset)
        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 保存最佳模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'New best model saved with accuracy: {best_acc:.4f}')

    print(f'Best val Acc: {best_acc:4f}')

# --- 5. 主函数 ---
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, valid_loader = get_dataloaders()
    model = get_model().to(device)

    print("Starting training...")
    train_model(model, train_loader, valid_loader, device)
    print("Training complete.")

