"""
LEAM 시스템 종합 성능 평가.

보고서 7절 기준:
  E_total = alpha * E_gaze + (1-alpha) * E_audio    (학습환경 적응 종합)
  rtC_total = sum(w_i * P_i)                        (실시간 집중도 파악 종합)

사용법:
    python eval_all.py
"""
import json
from pathlib import Path


def load_json(path: str):
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def evaluate_all():
    results_dir = Path(__file__).parent

    # Load individual results
    zone_results = load_json(results_dir / "results_learning_zone.json")
    audio_results = load_json(results_dir / "results_audio.json")
    phone_results = load_json(results_dir / "results_phone.json")
    drowsy_results = load_json(results_dir / "results_drowsiness.json")

    print("=" * 70)
    print("  LEAM System - Comprehensive Performance Report")
    print("=" * 70)

    # === Section 1: Environment Adaptation (E_total) ===
    print("\n[1] 사용자 학습환경 적응도 평가 (E_total)")
    print("-" * 50)

    e_gaze = 0
    if zone_results:
        e_gaze = zone_results.get("mean_iou", 0)
        print(f"  E_gaze (Learning Zone IoU):    {e_gaze:.4f}")
        dice = zone_results.get("mean_dice", 0)
        print(f"  보조지표 (Dice):               {dice:.4f}")
    else:
        print("  E_gaze: 평가 결과 없음 (eval_learning_zone.py 실행 필요)")

    e_audio = 0
    if audio_results:
        e_audio = audio_results.get("accuracy", 0)
        macro_f1 = audio_results.get("macro_f1", 0)
        print(f"  E_audio (Env Classification):  {e_audio:.4f}")
        print(f"  보조지표 (Macro F1):           {macro_f1:.4f}")
    else:
        print("  E_audio: 평가 결과 없음 (eval_audio_classifier.py 실행 필요)")

    alpha = 0.7  # 시선 적응 가중치 (보고서 기준)
    e_total = alpha * e_gaze + (1 - alpha) * e_audio
    print(f"\n  E_total = {alpha}*E_gaze + {1-alpha}*E_audio = {e_total:.4f}")

    # === Section 2: Real-time Concentration (rtC_total) ===
    print(f"\n[2] 실시간 집중도 파악 정확도 (rtC_total)")
    print("-" * 50)

    components = {}

    # Face detection (MediaPipe - generally very high, benchmark only)
    components["face_detection"] = {"name": "얼굴 탐지 (MediaPipe)", "weight": 0.1}
    print(f"  얼굴 탐지: MediaPipe FaceMesh (사전학습 모델, AP~0.95+)")
    components["face_detection"]["score"] = 0.95  # MediaPipe benchmark

    # Drowsiness
    if drowsy_results:
        best = drowsy_results.get("best", {})
        acc = best.get("accuracy", 0)
        f1 = best.get("f1", 0)
        print(f"  수면/졸음 (EAR):  Acc={acc:.4f}  F1={f1:.4f}")
        components["drowsiness"] = {"name": "수면/졸음", "weight": 0.25, "score": acc}
    else:
        print(f"  수면/졸음: 평가 결과 없음")
        components["drowsiness"] = {"name": "수면/졸음", "weight": 0.25, "score": 0}

    # Phone detection
    if phone_results:
        ap50 = phone_results.get("mAP50", 0)
        prec = phone_results.get("precision", 0)
        rec = phone_results.get("recall", 0)
        print(f"  휴대폰 탐지:      AP50={ap50:.4f}  P={prec:.4f}  R={rec:.4f}")
        components["phone"] = {"name": "휴대폰 탐지", "weight": 0.25, "score": ap50}
    else:
        print(f"  휴대폰 탐지: 평가 결과 없음")
        components["phone"] = {"name": "휴대폰 탐지", "weight": 0.25, "score": 0}

    # Gaze deviation (algorithm-based, evaluated via zone accuracy)
    gaze_score = e_gaze  # Zone accuracy directly affects gaze judgment
    print(f"  시선 이탈:        Zone기반 (IoU={gaze_score:.4f})")
    components["gaze"] = {"name": "시선 이탈", "weight": 0.25, "score": gaze_score}

    # Noise
    noise_score = e_audio
    print(f"  소음 파악:        Acc={noise_score:.4f}")
    components["noise"] = {"name": "소음 파악", "weight": 0.15, "score": noise_score}

    rtc_total = sum(c["weight"] * c["score"] for c in components.values())
    print(f"\n  rtC_total = sum(w_i * P_i) = {rtc_total:.4f}")

    # === Summary ===
    print(f"\n{'=' * 70}")
    print(f"  [종합 결과]")
    print(f"    E_total  (환경 적응 성능):          {e_total:.4f}")
    print(f"    rtC_total (실시간 집중도 판단 성능): {rtc_total:.4f}")
    print(f"{'=' * 70}")

    summary = {
        "E_total": round(e_total, 4),
        "E_gaze": round(e_gaze, 4),
        "E_audio": round(e_audio, 4),
        "alpha": alpha,
        "rtC_total": round(rtc_total, 4),
        "rtC_components": {k: {"score": round(v["score"], 4), "weight": v["weight"]}
                           for k, v in components.items()},
    }

    output_path = results_dir / "results_comprehensive.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    evaluate_all()
