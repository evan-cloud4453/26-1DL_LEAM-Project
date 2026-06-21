import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from app.services.learning_zone import LearningZoneGenerator

np.random.seed(42)
gen = LearningZoneGenerator()
t = 0.0
# 시나리오: 모니터(0.3,0.4) 응시 60%, 책(0.7,0.75) 응시 30%, 나머지 saccade 노이즈
for cycle in range(30):
    for _ in range(60):  # 모니터 fixation 2초
        gen.add_gaze_point(0.3 + np.random.normal(0, 0.02), 0.4 + np.random.normal(0, 0.02), t); t += 0.033
    for k in range(6):   # saccade 0.2초
        gen.add_gaze_point(0.3 + k * 0.07, 0.4 + k * 0.06, t); t += 0.033
    for _ in range(30):  # 책 fixation 1초
        gen.add_gaze_point(0.7 + np.random.normal(0, 0.02), 0.75 + np.random.normal(0, 0.02), t); t += 0.033

result = gen.generate_zones()
print('method:', result['method'])
print('zones (k):', result['k'])
print('fixations:', result['fixation_count'], '/ points:', result['gaze_point_count'])
for z in result['zones']:
    if z['type'] == 'study':
        print(f"  zone center=({z['center_x']:.2f},{z['center_y']:.2f}) "
              f"bbox=({z['x_min']:.2f}-{z['x_max']:.2f}, {z['y_min']:.2f}-{z['y_max']:.2f}) "
              f"area={z['area_ratio']:.3f}")
zones = result['zones']
print('모니터 (0.3,0.4) in-zone:', gen.is_in_study_zone(0.3, 0.4, zones))
print('책 (0.7,0.75) in-zone:', gen.is_in_study_zone(0.7, 0.75, zones))
print('빈공간 (0.9,0.1) in-zone:', gen.is_in_study_zone(0.9, 0.1, zones))
print('빈공간 (0.1,0.9) in-zone:', gen.is_in_study_zone(0.1, 0.9, zones))
