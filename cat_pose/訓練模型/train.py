import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import time
import glob
from tqdm import tqdm
import sys
import numpy as np # 新增: 用於處理損失比較
import matplotlib.pyplot as plt  # 新增：繪圖用


# 確保從 model.py 導入正確的模型名稱和輸入尺寸
# ⚠️ 假設您有一個 model.py 檔案，其中定義了 FinalSmallCNN 和 INPUT_IMAGE_SIZE
try:
    from model import BalancedMediumCNN as get_model, INPUT_IMAGE_SIZE
except ImportError:
    print("❌ 錯誤: 無法從 model.py 導入 FinalSmallCNN 或 INPUT_IMAGE_SIZE。")
    print("請確保 model.py 存在並定義了所需的類別和常數。")
    sys.exit(1)

# === 資料夾路徑設定與配置 ===
CONFIG = {
    'train_dir': r'C:\tinycnn\120', # 請確認您的訓練集路徑
    'valid_dir': r'C:\tinycnn\121', # 請確認您的驗證集路徑
    'img_size': (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
    'num_classes': None,
    'class_mapping': None,
    'batch_size': 64,
    'num_workers': 2,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu', # 📌 GPU 判斷
    'model_path': 'final_small_cnn_best_loss.pth', # 💡 模型保存名稱改為追蹤損失

    # === Early Stopping 參數：監控損失 ===
    'patience': 7, # 💡 連續 N 個 Epoch 驗證**損失**未改善就停止
    'min_delta': 1e-4, # 💡 損失需下降至少 0.0001 才算改善
}

# === 數據集轉換 (Data Augmentation & Preprocessing) ===
transform_train = transforms.Compose([
    transforms.Resize(CONFIG['img_size']),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize(CONFIG['img_size']),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === 自定義 Dataset 類別 (保持不變) ===
class YoloClassificationDataset(Dataset):
    """
    處理 YOLO Object Detection 格式數據集，但只提取 class_id 用於 Classification 訓練。
    (此處省略與您原始程式碼重複的 __init__、_get_image_files 等方法，假設其功能正常)
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = self._get_image_files()
        self.all_classes = self._get_all_classes()

        sorted_classes = sorted(list(self.all_classes))
        self.class_to_idx = {cls_id: idx for idx, cls_id in enumerate(sorted_classes)}
        self.class_mapping = {idx: cls_id for cls_id, idx in self.class_to_idx.items()}
        self.num_classes = len(self.all_classes)

        self.class_image_counts = self._count_class_image_files()

        if not self.image_files:
            print(f"⚠️ 警告: 在 {root_dir} 中找不到任何圖片檔案。")
            
        print(f"Dataset: {root_dir}")
        print(f"  找到圖片數: {len(self.image_files)}")
        print(f"  找到類別數: {self.num_classes} (原始ID: {sorted_classes})")
        print(f"  映射: {self.class_mapping}")
        print("  圖片數量統計 (原始ID -> 圖片數):")
        for cls_id, count in sorted(self.class_image_counts.items()):
            print(f"    - Class {cls_id}: {count} 張圖片")
        print("-" * 30)

    def _get_image_files(self):
        image_dir = os.path.join(self.root_dir, 'images')
        if not os.path.isdir(image_dir): return []

        files = glob.glob(os.path.join(image_dir, '*.[jp][pn]g')) + \
                glob.glob(os.path.join(image_dir, '*.bmp'))
        return files

    def _get_all_classes(self):
        label_dir = os.path.join(self.root_dir, 'labels')
        all_classes = set()
        if not os.path.isdir(label_dir): return all_classes

        for img_path in self.image_files:
            label_filename = os.path.basename(img_path).rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(label_dir, label_filename)

            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            class_id = int(line.split(' ')[0])
                            all_classes.add(class_id)
                except Exception:
                    pass
        return all_classes

    def _count_class_image_files(self):
        """統計每個原始類別 ID 包含多少張圖片。"""
        label_dir = os.path.join(self.root_dir, 'labels')
        counts = {}

        for img_path in self.image_files:
            label_filename = os.path.basename(img_path).rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(label_dir, label_filename)

            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            class_id = int(line.split(' ')[0])
                            counts[class_id] = counts.get(class_id, 0) + 1
                except:
                    pass
        return counts

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')

        label_filename = os.path.basename(img_path).rsplit('.', 1)[0] + '.txt'
        label_path = os.path.join(self.root_dir, 'labels', label_filename)

        label_id = 0

        try:
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    original_class_id = int(line.split(' ')[0])
                    label_id = self.class_to_idx[original_class_id]
                else:
                    raise ValueError("Label file is empty or malformed.")
        except Exception:
            pass 

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label_id, dtype=torch.long)

        return image, label


# === 輔助類別：Early Stopping 實現 (修改為監控損失) ===
class EarlyStopping:
    """早停法實作：監控**驗證損失**，連續未改善 N 次則停止訓練。"""
    def __init__(self, patience=7, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        if self.mode == 'min':
            self.monitor_op = np.less # 損失：越小越好
            self.delta = -self.min_delta # 改善門檻為負數 (score < best_score - min_delta)
        elif self.mode == 'max':
            self.monitor_op = np.greater # 準確率：越大越好
            self.delta = self.min_delta

    def __call__(self, score): # 接收的是當前驗證損失 (val_loss)
        
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            return False

        # 判斷是否有實質改善
        if self.monitor_op(score, self.best_score + self.delta):   # min 模式時，只有當 val_loss < best_score - min_delta 才視為改善
            self.best_score = score # 更新最低損失
            self.counter = 0
            return False # 損失仍在改善，不停止
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"🚨 Early Stopping 觸發! (連續 {self.patience} 輪 Val Loss 未改善)")
            return self.early_stop

# === 訓練主函數 (修改為監控驗證損失) ===
def train_model():
    print(f"使用設備: {CONFIG['device']}")

    total_start_time = time.time()

    # 1. 數據加載
    try:
        train_dataset = YoloClassificationDataset(CONFIG['train_dir'], transform=transform_train)
        val_dataset = YoloClassificationDataset(CONFIG['valid_dir'], transform=transform_val)

        if train_dataset.num_classes != val_dataset.num_classes:
            print(f"❌ 錯誤: 訓練集類別數 ({train_dataset.num_classes}) 與驗證集類別數 ({val_dataset.num_classes}) 不一致。")
            return

        CONFIG['num_classes'] = train_dataset.num_classes
        CONFIG['class_mapping'] = train_dataset.class_mapping

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                                     shuffle=True, num_workers=CONFIG['num_workers'])
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                                 shuffle=False, num_workers=CONFIG['num_workers'])

    except Exception as e:
        print(f"⚠️ 數據集初始化失敗: {e}")
        return

    # 2. 模型與訓練設定
    device = torch.device(CONFIG['device'])
    model = get_model(num_classes=CONFIG['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    early_stopper = EarlyStopping(patience=CONFIG['patience'], min_delta=CONFIG['min_delta'], mode='min')

    best_val_loss = float('inf')
    best_val_accuracy = 0.0

    # === 新增：儲存損失紀錄 ===
    train_losses = []
    val_losses = []

    # 3. 訓練循環
    print("-" * 60)
    print(f"開始訓練 (最多 {CONFIG['num_epochs']} 輪)")
    print("-" * 60)

    for epoch in range(CONFIG['num_epochs']):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1:02d}/{CONFIG['num_epochs']} (Train)"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_dataset)

        # === 驗證階段 ===
        model.eval()
        val_running_loss = 0.0
        correct, total = 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss = criterion(outputs, labels)
                val_running_loss += val_loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / len(val_dataset)
        epoch_val_acc = 100 * correct / total
        scheduler.step()

        # === 儲存損失 ===
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        # === 顯示進度 ===
        print(f"Epoch {epoch+1:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        # === 保存最佳模型 ===
        if epoch_val_loss < best_val_loss - CONFIG['min_delta']:
            best_val_loss = epoch_val_loss
            best_val_accuracy = epoch_val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'num_classes': CONFIG['num_classes'],
                'class_mapping': CONFIG['class_mapping'],
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_accuracy,
                'epoch': epoch
            }, CONFIG['model_path'])
            print(f"✨ 模型保存! Val Loss 降至 {best_val_loss:.4f} (Acc: {best_val_accuracy:.2f}%)。")

        if early_stopper(epoch_val_loss):
            break

    # === 繪製收斂圖 ===
    plt.figure(figsize=(8,5))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.title('Training & Validation Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('loss_curve.png')


    # 📌 訓練完成
    total_time_spent = time.time() - total_start_time
    print("-" * 40)
    print("訓練完成")
    print(f"🎉 總訓練時間: {total_time_spent:.2f} 秒")
    print(f"最佳模型保存時的 Val Loss: {best_val_loss:.4f}")
    print(f"最佳模型保存時的 Val Accuracy: {best_val_accuracy:.2f}%")
    print(f"模型保存在: {CONFIG['model_path']}")
    

if __name__ == '__main__':
    train_model()