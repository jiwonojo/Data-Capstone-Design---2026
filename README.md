# 💳 Leakage-Free FDS: 행동 기반 파생 변수를 활용한 이상거래 탐지 모델 고도화 및 한계 분석

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-Advanced-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-red.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-success.svg)

> **데이터 사이언스 캡스톤디자인 프로젝트** > 
> **팀원:** 강지원(기획/아키텍처), 오은서(데이터 전처리/FE), 최소현(모델링/XAI)

## 📌 Project Overview
신용카드 이상거래 탐지(FDS, Fraud Detection System) 모델링 과정에서 빈번하게 발생하는 **데이터 누수(Data Leakage)** 현상을 식별 및 차단하고, 실제 사용자의 행동 프로파일링(Behavioral Profiling)에 기반한 **순수 파생 변수(Feature Engineering)** 만으로 탐지 모델을 구축하는 프로젝트입니다. 

초기 99%의 가짜 성능(Leakage)을 버리고, Autoencoder 기반의 하이브리드 확장 및 Stacking 앙상블을 적용하여 모델을 고도화했습니다. 최종적으로 SHAP 분석을 통해 모델의 한계가 아닌 **'데이터 자체의 사기 예측 신호(Signal) 결핍'이라는 Root Cause를 정량적으로 증명**해 냈습니다.

---

## 🚨 1. Data Leakage Issue & Pivot
초기 베이스라인 모델 학습 결과 ROC-AUC 0.99라는 비정상적인 성능이 도출되었습니다. Feature Importance 분석 결과, 특정 변수들이 예측을 지배(99% 이상)하는 것을 확인했습니다.

* **Target Leakage:** `Risk_Score` (시스템이 최종 산출해야 할 결과값이 입력으로 들어감)
* **Time-Series Leakage:** `Failed_Transaction_Count_7d` (현재 결제 시점에서는 알 수 없는 미래의 실패 건수 누적)

**💡 Action:** 해당 누수 변수들을 데이터셋에서 완전히 삭제. 성능은 ROC-AUC 0.50 수준으로 붕괴하였으나, 이를 **'정답지를 뺏긴 상태에서의 진짜 베이스라인'**으로 삼고 아키텍처를 전면 재설계했습니다.

---

## 🛠 2. Feature Engineering (행동 기반 프로파일링)
정답지가 사라진 상황에서 모델에 사기 탐지 신호(Signal)를 제공하기 위해, 3가지 도메인 지식을 기반으로 **총 9개의 파생 변수**를 신규 설계했습니다. (Pandas `shift`, `cumsum` 등을 활용해 미래 데이터 참조 방지)

1.  **금액 및 잔고 (Bust-out 사기 타겟팅)**
    * `Balance_to_Amount`: 잔액 대비 거래액 비율
    * `Amount_vs_User_Avg`: 본인 과거 평균 결제액 대비 비율
    * `Amount_vs_7d_Avg`: 최근 7일 평균 대비 이탈률
2.  **시공간 및 속도 (Impossible Travel 타겟팅)**
    * `Velocity`: 직전 결제 위치와의 거리 및 시간차를 이용한 물리적 이동 속도
    * `Distance_vs_User_Avg`: 본인 평균 이동 반경 대비 현재 거리
    * `Transaction_Interval`: 단기 연속 거래 간격
3.  **환경 및 맥락 결합 (Contextual Anomaly)**
    * `Weekend_Night`: 주말 심야 취약 시간대 여부
    * `Is_New_Device`: 과거 결제 이력이 없는 신규 기기 여부
    * `High_Daily_Count`: 일일 과다 결제 발생 여부

---

## ⚙️ 3. Modeling & Hybrid Architecture
단일 모델의 한계를 극복하고 예측 성능을 끌어올리기 위해 다양한 확장을 시도했습니다.

* **Hyperparameter Tuning:** `RandomizedSearchCV` 및 Stratified 5-Fold CV를 적용하여 XGBoost, LightGBM 파라미터 최적화.
* **Autoencoder (비지도 학습 결합):** 정상 데이터(Label=0)만을 활용해 오토인코더를 학습시킨 후, 사기 데이터 입력 시 발생하는 **재구성 오차(Reconstruction Error)**를 새로운 Feature로 추가.
* **Stacking Ensemble:** XGBoost, LightGBM, Random Forest를 Base Learner로 두고, Logistic Regression을 Meta Model로 사용하는 스태킹 아키텍처 적용.

---

## 📊 4. Results & Root Cause Analysis (SHAP)
모든 고도화 및 앙상블 기법을 동원했음에도 불구하고, 모델의 최종 성능은 **ROC-AUC 0.50, PR-AUC ~0.32** 수준에서 정체되었습니다.

### 🔍 왜 성능이 오르지 않았는가? (SHAP Analysis)
이 현상의 근본 원인(Root Cause)을 밝히기 위해 SHAP Value를 추출하여 분석했습니다.
* Tree 모델의 Feature Importance에서는 우리가 만든 파생 변수(`Amount_vs_User_Avg`, `Velocity` 등)가 최상위권에 올랐습니다. (자주 분기에 사용됨)
* **그러나 SHAP Summary Plot 분석 결과, 대부분 피처들의 SHAP 값이 '0' 부근에 조밀하게 뭉쳐 있었습니다.**


> **💡 최종 결론:** > 누수 변수(정답지)가 제거된 현재의 순수 거래 데이터셋 자체가 사기와 정상을 명확히 구분 지을 수 있는 **'강한 정보량(Signal/Label Separation)'을 근본적으로 결여하고 있음**을 시각적, 통계적으로 입증한 결과입니다. 실무 FDS 시스템 구축 시, 단순한 알고리즘 고도화보다 양질의 다차원 피처(Rolling 이력 등) 수집이 필수적임을 의미합니다.

---

## 💻 Tech Stack
- **Data Manipulation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`, `xgboost`, `lightgbm`
- **Deep Learning (Autoencoder):** `TensorFlow` / `Keras` (또는 PyTorch)
- **Explainable AI (XAI):** `shap`
- **Visualization:** `matplotlib`, `seaborn`
