from ultralytics import YOLO
import torch
import json
from datetime import datetime
from pathlib import Path


def train_cat_pose(model_path, data_yaml, epochs=200, log_dir="training_logs"):
    """
    YOLOv8 Pose - 貓咪 17/24 Keypoints 專用訓練設定
    適合小資料集 (~1200 張)
    """
    
    # 記錄訓練開始時間
    start_time = datetime.now()
    start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # 建立日誌目錄
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 載入模型
    model = YOLO(model_path)
    
    # 裝置選擇
    device = 0 if torch.cuda.is_available() else "cpu"
    
    # 訓練參數設定（直接定義在這裡）
    train_config = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": 640,
        "batch": 16,
        "device": device,
        "workers": 0,
        "cache": "ram",
        "optimizer": "AdamW",
        "lr0": 0.0005,
        "lrf": 0.001,
        "cos_lr": True,
        "amp": True,
        "patience": 30,
        "degrees": 15,
        "translate": 0.1,
        "scale": 0.1,
        "shear": 0.0,
       "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.4,
        "mosaic": 0,
        "mixup": 0,
        "copy_paste": 0.0,
    }
    
    print(f"開始訓練時間: {start_time_str}")
    print(f"使用裝置: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # 執行訓練
    results = model.train(**train_config)
    
    # 記錄結束時間
    end_time = datetime.now()
    end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    duration = end_time - start_time
    
    # 收集訓練結果指標
    try:
        metrics_file = Path(results.save_dir) / "results.csv"
        if metrics_file.exists():
            import pandas as pd
            df = pd.read_csv(metrics_file)
            
            # 找出最佳 epoch
            best_epoch_idx = df['metrics/mAP50-95(B)'].idxmax()
            best_row = df.iloc[best_epoch_idx]
            
            best_metrics = {
                "epoch": int(best_row['epoch']),
                "box_loss": float(best_row['train/box_loss']),
                "pose_loss": float(best_row['train/pose_loss']),
                "kobj_loss": float(best_row['train/kobj_loss']),
                "cls_loss": float(best_row['train/cls_loss']),
                "dfl_loss": float(best_row['train/dfl_loss']),
                "mAP50(B)": float(best_row['metrics/mAP50(B)']),
                "mAP50-95(B)": float(best_row['metrics/mAP50-95(B)']),
                "mAP50(P)": float(best_row['metrics/mAP50(P)']),
                "mAP50-95(P)": float(best_row['metrics/mAP50-95(P)']),
            }
        else:
            best_metrics = {"note": "results.csv not found"}
    except Exception as e:
        best_metrics = {"error": str(e)}
    
    # 建立完整日誌
    log_entry = {
        "training_info": {
            "start_time": start_time_str,
            "end_time": end_time_str,
            "duration_seconds": round(duration.total_seconds(), 2),
            "duration_formatted": str(duration).split('.')[0],  # 移除毫秒
            "model_type": str(model_path),
            "data_yaml": str(data_yaml),
        },
        "training_parameters": train_config,
        "system_info": {
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "pytorch_version": torch.__version__,
            "device_used": str(device),
        },
        "results": {
            "save_directory": str(results.save_dir),
            "best_model": str(Path(results.save_dir) / "weights" / "best.pt"),
            "last_model": str(Path(results.save_dir) / "weights" / "last.pt"),
            "best_epoch_metrics": best_metrics,
        }
    }
    
    # 儲存 JSON 日誌
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    log_filename = f"training_log_{timestamp}.json"
    log_file = log_path / log_filename
    
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)
    
    # 儲存易讀的文字日誌
    txt_log_file = log_path / f"training_log_{timestamp}.txt"
    
    with open(txt_log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("YOLOv8 貓咪姿態估計訓練日誌\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("📅 時間資訊\n")
        f.write("-" * 80 + "\n")
        f.write(f"開始時間: {start_time_str}\n")
        f.write(f"結束時間: {end_time_str}\n")
        f.write(f"訓練時長: {str(duration).split('.')[0]}\n\n")
        
        f.write("🔧 訓練參數\n")
        f.write("-" * 80 + "\n")
        f.write(f"模型: {model_path}\n")
        f.write(f"數據集: {data_yaml}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {train_config['batch']}\n")
        f.write(f"Image Size: {train_config['imgsz']}\n")
        f.write(f"Optimizer: {train_config['optimizer']}\n")
        f.write(f"Learning Rate (初始): {train_config['lr0']}\n")
        f.write(f"Learning Rate (最終): {train_config['lrf']}\n")
        f.write(f"Cosine LR: {train_config['cos_lr']}\n")
        f.write(f"AMP: {train_config['amp']}\n")
        f.write(f"Patience: {train_config['patience']}\n\n")
        
        f.write("🎨 數據增強參數\n")
        f.write("-" * 80 + "\n")
        f.write(f"Degrees: {train_config['degrees']}\n")
        f.write(f"Translate: {train_config['translate']}\n")
        f.write(f"Scale: {train_config['scale']}\n")
        f.write(f"Shear: {train_config['shear']}\n")
        f.write(f"FlipLR: {train_config['fliplr']}\n")
        f.write(f"FlipUD: {train_config['flipud']}\n")
        f.write(f"HSV_H: {train_config['hsv_h']}\n")
        f.write(f"HSV_S: {train_config['hsv_s']}\n")
        f.write(f"HSV_V: {train_config['hsv_v']}\n")
        f.write(f"Mosaic: {train_config['mosaic']}\n")
        f.write(f"Mixup: {train_config['mixup']}\n\n")
        
        f.write("💻 系統資訊\n")
        f.write("-" * 80 + "\n")
        f.write(f"CUDA 可用: {torch.cuda.is_available()}\n")
        if torch.cuda.is_available():
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
        f.write(f"PyTorch 版本: {torch.__version__}\n")
        f.write(f"訓練裝置: {device}\n\n")
        
        f.write("📊 訓練結果\n")
        f.write("-" * 80 + "\n")
        f.write(f"儲存路徑: {results.save_dir}\n")
        f.write(f"最佳模型: {results.save_dir}/weights/best.pt\n")
        f.write(f"最終模型: {results.save_dir}/weights/last.pt\n\n")
        
        if isinstance(best_metrics, dict) and "epoch" in best_metrics:
            f.write(f"🏆 最佳表現 (Epoch {best_metrics['epoch']})\n")
            f.write("-" * 80 + "\n")
            f.write(f"Box mAP50: {best_metrics['mAP50(B)']:.4f}\n")
            f.write(f"Box mAP50-95: {best_metrics['mAP50-95(B)']:.4f}\n")
            f.write(f"Pose mAP50: {best_metrics['mAP50(P)']:.4f}\n")
            f.write(f"Pose mAP50-95: {best_metrics['mAP50-95(P)']:.4f}\n")
            f.write(f"\nLoss 數值:\n")
            f.write(f"  Box Loss: {best_metrics['box_loss']:.4f}\n")
            f.write(f"  Pose Loss: {best_metrics['pose_loss']:.4f}\n")
            f.write(f"  Kobj Loss: {best_metrics['kobj_loss']:.4f}\n")
            f.write(f"  Cls Loss: {best_metrics['cls_loss']:.4f}\n")
            f.write(f"  DFL Loss: {best_metrics['dfl_loss']:.4f}\n")
    
    print(f"\n{'='*80}")
    print(f"✅ 訓練完成！")
    print(f"⏱️  訓練時長: {str(duration).split('.')[0]}")
    print(f"📁 最佳模型: {results.save_dir}/weights/best.pt")
    print(f"📝 日誌已儲存:")
    print(f"   • JSON: {log_file}")
    print(f"   • TXT:  {txt_log_file}")
    print(f"{'='*80}\n")
    
    return results


if __name__ == "__main__":
    
    model_path = "yolo11s-pose.pt"
    data_yaml = r"C:\cat_pose\訓練模型\data.yaml"
    log_directory = r"C:\cat_pose\訓練模型\training_logs"
    
    train_cat_pose(
        model_path=model_path, 
        data_yaml=data_yaml, 
        epochs=300,
        log_dir=log_directory
    )