"""
YOLOv8n fine-tuning for phone detection.

Dataset: AI Hub "Small object detection을 위한 이미지 데이터"
         + Custom dataset with phone-in-hand study scenarios

Usage:
    python train_phone_detector.py --data_dir ./datasets/phone --epochs 50

Directory structure expected:
    datasets/phone/
        images/
            train/
            val/
        labels/
            train/
            val/
        data.yaml
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def create_data_yaml(data_dir: Path) -> Path:
    yaml_path = data_dir / "data.yaml"
    # 한글 경로 손상 방지: 슬래시 경로 + UTF-8 저장 (항상 재생성)
    abs_path = data_dir.resolve().as_posix()
    content = f"""
path: {abs_path}
train: images/train
val: images/val

nc: 1
names: ['cell_phone']
"""
    yaml_path.write_text(content.strip(), encoding="utf-8")
    return yaml_path


def train(data_dir: str, epochs: int = 50, imgsz: int = 640, batch: int = 16):
    data_path = Path(data_dir)
    yaml_path = create_data_yaml(data_path)

    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name="leam_phone_detector",
        project="runs/detect",
        patience=10,
        save=True,
        device="0",  # GPU; use "cpu" if no GPU
        workers=4,
        # Transfer learning: freeze backbone for first N epochs
        freeze=10,
    )

    # 실제 저장 위치를 trainer에서 동적으로 가져온다 (경로 doubling/증분 대응)
    best_model_path = Path(model.trainer.save_dir) / "weights" / "best.pt"
    output_path = Path(__file__).parent.parent / "weights" / "yolov8n_phone.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if best_model_path.exists():
        import shutil
        shutil.copy(best_model_path, output_path)
        print(f"Best model saved to {output_path}")
    else:
        print(f"[warn] best.pt를 찾지 못함: {best_model_path}")

    return results


def validate(weights_path: str, data_dir: str):
    model = YOLO(weights_path)
    data_path = Path(data_dir)
    yaml_path = create_data_yaml(data_path)

    metrics = model.val(data=str(yaml_path))
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8n for phone detection")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--weights", type=str, default=None)
    args = parser.parse_args()

    if args.validate and args.weights:
        validate(args.weights, args.data_dir)
    else:
        train(args.data_dir, args.epochs, args.imgsz, args.batch)
