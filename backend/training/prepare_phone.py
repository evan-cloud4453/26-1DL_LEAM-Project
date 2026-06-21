"""
AI Hub "Small object detection을 위한 이미지 데이터" → YOLO 휴대폰 탐지 데이터셋.

전자기기 49종 중 '휴대폰'(class_id=126)만 추출하여 단일 클래스(cell_phone)로 변환.
Superb AI(COCO 유사) JSON의 bbox [x,y,w,h] 픽셀 → YOLO 정규화 [cx,cy,w,h].

  datasets/phone/
    images/{train,val}/*.jpg
    labels/{train,val}/*.txt   (각 줄: 0 cx cy w h)

Usage:
    python training/prepare_phone.py
    python training/prepare_phone.py --val_ratio 0.2 --seed 42
"""
import argparse
import glob
import json
import os
import random
import shutil
from pathlib import Path

from PIL import Image

PHONE_CLASS_ID = 126  # AI Hub '휴대폰'

DEFAULT_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "Small object detection을 위한 이미지 데이터"
)


def build_image_index(root: Path) -> dict[str, Path]:
    idx = {}
    for p in glob.glob(str(root / "1.원천데이터" / "**" / "*.jpg"), recursive=True):
        idx[Path(p).stem] = Path(p)
    return idx


def collect_phone_samples(root: Path, img_index: dict[str, Path]):
    """[(image_path, [(x,y,w,h), ...]), ...] — 휴대폰 박스가 있는 이미지만."""
    samples = []
    label_files = glob.glob(str(root / "2.라벨링데이터" / "**" / "*.json"), recursive=True)
    for f in label_files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        boxes = [a["bbox"] for a in d.get("annotations", [])
                 if a.get("category_id") == PHONE_CLASS_ID and a.get("bbox")]
        if not boxes:
            continue
        stem = Path(f).stem
        img_path = img_index.get(stem)
        if img_path is None:
            continue
        samples.append((img_path, boxes))
    return samples


def to_yolo_lines(boxes, W, H) -> list[str]:
    lines = []
    for x, y, w, h in boxes:
        cx = (x + w / 2) / W
        cy = (y + h / 2) / H
        nw, nh = w / W, h / H
        # 경계 클램프
        cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
        nw, nh = min(max(nw, 0), 1), min(max(nh, 0), 1)
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def write_split(samples, out_root: Path, split: str) -> int:
    img_dir = out_root / "images" / split
    lbl_dir = out_root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path, boxes in samples:
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            continue
        lines = to_yolo_lines(boxes, W, H)
        if not lines:
            continue
        stem = img_path.stem
        shutil.copy2(img_path, img_dir / f"{stem}.jpg")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="AI Hub → YOLO 휴대폰 데이터셋 변환")
    ap.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "datasets" / "phone"))
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root, out_root = Path(args.root), Path(args.out)
    print(f"데이터 루트: {root}")
    img_index = build_image_index(root)
    print(f"이미지 인덱싱: {len(img_index)}개")

    samples = collect_phone_samples(root, img_index)
    print(f"휴대폰 포함 이미지: {len(samples)}개")
    if not samples:
        print("휴대폰 샘플이 없습니다. 종료.")
        return

    random.seed(args.seed)
    random.shuffle(samples)
    n_val = int(len(samples) * args.val_ratio)
    val_s, train_s = samples[:n_val], samples[n_val:]

    if out_root.exists():
        shutil.rmtree(out_root)

    n_train = write_split(train_s, out_root, "train")
    n_val_w = write_split(val_s, out_root, "val")

    print("\n=== 변환 완료 ===")
    print(f"train: {n_train}개  /  val: {n_val_w}개")
    print(f"출력: {out_root}")
    print("다음: python training/train_phone_detector.py --data_dir ./datasets/phone --epochs 50")


if __name__ == "__main__":
    main()
