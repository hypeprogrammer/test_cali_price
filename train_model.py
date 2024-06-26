import numpy as np  # 수치 계산을 위한 라이브러리 임포트
import pandas as pd  # 데이터 처리를 위한 라이브러리 임포트
from sklearn.datasets import fetch_california_housing  # 캘리포니아 주택 가격 데이터셋 로드를 위한 모듈 임포트
from sklearn.model_selection import train_test_split  # 데이터셋 분할을 위한 모듈 임포트
from sklearn.linear_model import LinearRegression  # 선형 회귀 모델을 위한 모듈 임포트
from sklearn.metrics import mean_squared_error  # 모델 평가를 위한 평균 제곱 오차(MSE) 계산 모듈 임포트
import joblib  # 모델 저장 및 로드를 위한 모듈 임포트

# 캘리포니아 주택 가격 데이터셋 로드
california = fetch_california_housing()
X = california.data  # 피처 데이터 설정
y = california.target  # 타겟 값 설정

# 데이터를 학습용과 테스트용으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 훈련
model = LinearRegression()
model.fit(X_train, y_train)  # 모델 학습

# 모델 평가
y_pred = model.predict(X_test)  # 테스트 데이터를 사용하여 예측 수행
mse = mean_squared_error(y_test, y_pred)  # 평균 제곱 오차 계산
print(f"Mean Squared Error: {mse}")  # MSE 출력

# 모델 저장
joblib.dump(model, 'california_housing_model.pkl')  # 모델을 'california_housing_model.pkl' 파일로 저장
