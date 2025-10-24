# bigcon_2025

제13회 2025 빅콘테스트 AI데이터 분석분야: 
우리 동네 가맹점, 위기 신호를 미리 잡아라!


1. 프로젝트 개요
본 프로젝트는 2025 BigCon 데이터 분석 대회 출품작으로, 대회 제공 데이터(마스터 테이블, 추정 매출, 임대 시세) 및 공공 데이터(물가지수)를 기반으로 생성한 6종의 파생 지표를 활용하여, 서울시 성동구 소상공인 가맹점의 3개월/6개월 후 영업 위험도(매출 하락)를 예측하는 LightGBM 모델입니다.


2. 필요 환경
본 코드는 Python 3.11.9 (64-bit) 환경에서 정상 작동함을 확인하였습니다.
필요한 라이브러리는 requirements.txt 파일에 명시되어 있습니다.


3. 설치 및 실행 방법
터미널(cmd)에서 아래 명령어를 순서대로 입력하십시오.

# 가상 환경 생성 (py 명령어 사용 권장)
py -m venv venv

# 가상 환경 활성화 (Windows)
.\venv\Scripts\activate.bat

# requirements.txt 파일을 이용해 모든 라이브러리 설치
pip install -r requirements.txt

# merged_indices_monthly.parquet 파일 생성
python -m src.pipeline

# 3개월 및 6개월 모델 학습 후 AUC 결과 출력
python -X utf8 src/train.py


4. 기대 결과

터미널 출력: train.py 실행 시, 모델의 최적 성능이 출력됩니다.
Target=y_risk_3m 및 Target=y_risk_6m에 대한 최종 auc 점수가 출력됩니다.

파일 생성: 프로젝트 루트에 outputs 폴더가 자동 생성됩니다. outputs 폴더 내부에 다음 주요 파일이 생성됩니다.
merged_indices_monthly.parquet: 최종 학습 데이터셋
risk_lgbm_delta_3m.joblib: 3개월 예측 모델 파일
risk_lgbm_delta_6m.joblib: 6개월 예측 모델 파일



5. 첨언
모델을 실행하는 경우, 아래와 같은 requirements.txt를 이용하는 것을 권장합니다.

numpy==1.26.4
pandas==2.2.2
lightgbm==4.6.0
scikit-learn==1.7.2
pyarrow==15.0.2
scipy
joblib
threadpoolctl
python-dateutil
pytz
tzdata
