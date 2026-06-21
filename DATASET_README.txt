LEAM 프로젝트 — 사용 데이터셋 출처

데이터는 용량이 매우 크기 때문에 원본 대신 출처와 활용 내역을
남기는 방식으로 제출합니다. 두 데이터셋 모두 공개/오픈 데이터입니다.
----------------------------------------------------------------
[데이터 1] AI Hub — Small object detection을 위한 이미지 데이터
----------------------------------------------------------------
- URL    : https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=476
- 용도    : 휴대폰 탐지 모델(YOLOv8n) 전이학습
- 사용 범위: 원본의 전자기기 49종 중 '휴대폰(cell_phone)' 단일 클래스만 추출하여 사용
- 사용 수량: 학습 564장 / 검증 140장

----------------------------------------------------------------
[데이터 2] DCASE 2016 — TUT Acoustic Scenes 2016
----------------------------------------------------------------
- URL    : https://dcase.community/challenge2016/task-acoustic-scene-classification
- 용도    : 환경음 분류 모델(PANNs CNN10) 전이학습
- 사용 범위: 원본 15개 음향 장면 중 이동수단·순수 자연환경을 제외하고, 학습 맥락에
            맞춰 3개 클래스로 재매핑하여 사용
              · quiet         ← library, home
              · social_indoor ← cafe/restaurant, office, grocery store
              · outdoor       ← city center, residential area, park
- 데이터 분할: 공식 fold1 분할
- 사용 수량: 학습 468개 / 검증 156개 (5초 클립 기준)

감사합니다.