"""
DCASE 2016 (TUT Acoustic Scenes) → LEAM 3-class environment dataset.

15개 음향 장면을 학습 맥락의 3클래스로 매핑한다. 이동수단(bus/car/train/tram/
metro)과 순수 자연환경(beach/forest)은 "학습 중 카메라를 켤 일이 없는 환경"이라
제외한다 (보고서 환경 분류 = 적응 단계 축 A).

  quiet         : library, home                         (조용·집중적합)
  social_indoor : cafe/restaurant, office, grocery_store (웅성·대화 실내)
  outdoor       : city_center, residential_area, park    (도시형 야외 소음)

공식 fold1 split(train/evaluate)을 그대로 train/val로 사용한다 — 녹음 위치 기반
분리라 데이터 누수가 없다.

Usage:
    python training/prepare_dcase.py
    python training/prepare_dcase.py --dcase_root "<경로>" --out ./datasets/audio
"""
import argparse
import shutil
from pathlib import Path

# DCASE 장면 라벨 → LEAM 클래스 (없는 라벨은 제외)
SCENE_TO_CLASS = {
    "library": "quiet",
    "home": "quiet",
    "cafe/restaurant": "social_indoor",
    "office": "social_indoor",
    "grocery_store": "social_indoor",
    "city_center": "outdoor",
    "residential_area": "outdoor",
    "park": "outdoor",
    # 제외: beach, bus, car, forest_path, metro_station, train, tram
}

DEFAULT_DCASE_ROOT = (
    Path(__file__).resolve().parent.parent.parent
    / "data" / "TUT-acoustic-scenes-2016-development"
)


def build_audio_index(dcase_root: Path) -> dict[str, Path]:
    """basename(.wav) → 실제 전체 경로 (audio.1~8 폴더에 분산되어 있음)."""
    index: dict[str, Path] = {}
    for wav in dcase_root.glob("*.audio.*/**/audio/*.wav"):
        index[wav.name] = wav
    return index


def parse_fold(fold_file: Path) -> list[tuple[str, str]]:
    """fold txt → [(basename, scene_label), ...]"""
    entries = []
    for line in fold_file.read_text(encoding="utf-8").strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        rel_path, scene = parts[0].strip(), parts[1].strip()
        entries.append((Path(rel_path).name, scene))
    return entries


def copy_split(entries, audio_index, out_split: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    missing = 0
    for basename, scene in entries:
        cls = SCENE_TO_CLASS.get(scene)
        if cls is None:
            continue  # 제외 대상 장면
        src = audio_index.get(basename)
        if src is None:
            missing += 1
            continue
        dst_dir = out_split / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / basename)
        counts[cls] = counts.get(cls, 0) + 1
    if missing:
        print(f"  [warn] {missing}개 파일을 audio 폴더에서 찾지 못함")
    return counts


def main():
    parser = argparse.ArgumentParser(description="DCASE 2016 → LEAM 3-class 변환")
    parser.add_argument("--dcase_root", type=str, default=str(DEFAULT_DCASE_ROOT))
    parser.add_argument("--out", type=str,
                        default=str(Path(__file__).resolve().parent.parent / "datasets" / "audio"))
    parser.add_argument("--fold", type=int, default=1, help="사용할 공식 fold 번호 (1~4)")
    args = parser.parse_args()

    dcase_root = Path(args.dcase_root)
    out_root = Path(args.out)
    setup = (dcase_root / "TUT-acoustic-scenes-2016-development.meta"
             / "TUT-acoustic-scenes-2016-development" / "evaluation_setup")

    print(f"DCASE root: {dcase_root}")
    audio_index = build_audio_index(dcase_root)
    print(f"오디오 파일 인덱싱: {len(audio_index)}개\n")

    if out_root.exists():
        shutil.rmtree(out_root)

    train_entries = parse_fold(setup / f"fold{args.fold}_train.txt")
    val_entries = parse_fold(setup / f"fold{args.fold}_evaluate.txt")

    print("[train] 복사 중...")
    train_counts = copy_split(train_entries, audio_index, out_root / "train")
    print("[val] 복사 중...")
    val_counts = copy_split(val_entries, audio_index, out_root / "val")

    print("\n=== 변환 완료 ===")
    classes = sorted(set(SCENE_TO_CLASS.values()))
    print(f"{'class':<16}{'train':>8}{'val':>8}")
    for c in classes:
        print(f"{c:<16}{train_counts.get(c,0):>8}{val_counts.get(c,0):>8}")
    print(f"{'TOTAL':<16}{sum(train_counts.values()):>8}{sum(val_counts.values()):>8}")
    print(f"\n출력 위치: {out_root}")
    print("다음: python training/train_audio_classifier.py --data_dir ./datasets/audio")


if __name__ == "__main__":
    main()
