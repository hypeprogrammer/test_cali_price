from flask import Blueprint, request, render_template  # Flask 블루프린트, 요청, 템플릿 렌더링을 위한 모듈 임포트
from sklearn.datasets import fetch_california_housing  # 캘리포니아 주택가격 데이터셋 로드를 위한 모듈 임포트
from sklearn.model_selection import train_test_split  # 데이터셋 분할을 위한 모듈 임포트
from sklearn.linear_model import LinearRegression  # 선형 회귀 모델을 위한 모듈 임포트
from sklearn.preprocessing import StandardScaler  # 데이터 스케일링을 위한 모듈 임포트
from sklearn.metrics import mean_squared_error, r2_score  # 모델 평가를 위한 모듈 임포트
import pandas as pd  # 데이터 처리를 위한 라이브러리 임포트
import numpy as np  # 수치 계산을 위한 라이브러리 임포트

# 블루프린트 생성
predict_bp = Blueprint('predict_bp', __name__)

# 캘리포니아 주택가격 데이터 로드 및 모델 훈련
def load_data_and_train_model():
    california_housing = fetch_california_housing()  # 캘리포니아 주택가격 데이터셋 로드
    data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)  # 데이터프레임으로 변환
    data['MedHouseVal'] = california_housing.target  # 타겟 값을 데이터프레임에 추가

    X = data.drop('MedHouseVal', axis=1)  # 피처와 타겟 분리
    y = data['MedHouseVal']  # 타겟 값 설정

    # 타겟 값에 로그 변환 적용
    y = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 데이터를 학습용과 테스트용으로 분할

    # 피처 데이터에 스케일링 적용
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # 학습 데이터에 스케일링 적용 및 변환
    X_test = scaler.transform(X_test)  # 테스트 데이터에 동일한 스케일링 적용

    model = LinearRegression()  # 선형 회귀 모델 생성
    model.fit(X_train, y_train)  # 모델 훈련

    return model, X_train, X_test, y_train, y_test, scaler  # 모델과 데이터 반환

# 모델과 데이터 로드 및 훈련
model, X_train, X_test, y_train, y_test, scaler = load_data_and_train_model()

# 모델의 정확도 계산
y_pred = model.predict(X_test)  # 테스트 데이터를 사용하여 예측 수행
mse = mean_squared_error(y_test, y_pred)  # MSE(평균 제곱 오차) 계산
r2 = r2_score(y_test, y_pred)  # R2 점수 계산
accuracy = {
    'mse': mse,  # MSE 저장
    'r2': r2,  # R2 점수 저장
    'r2_percentage': r2 * 100  # R2 점수를 백분율로 변환하여 저장
}

# 예측 라우트 정의
@predict_bp.route('/', methods=['GET', 'POST'])
def predict_route():
    prediction = None  # 예측 결과 초기화
    if request.method == 'POST':  # POST 요청인 경우
        input_data = request.form  # 폼 데이터 가져오기
        input_df = pd.DataFrame([input_data.to_dict(flat=True)])  # 폼 데이터를 데이터프레임으로 변환
        input_df = input_df.astype(float)  # 데이터 타입을 float로 변환

        # 피처 데이터에 스케일링 적용
        input_df = scaler.transform(input_df)

        # 예측 수행
        prediction = model.predict(input_df)

        # 로그 변환된 타겟 값에 대해 역변환 수행
        prediction = np.expm1(prediction)[0]

    # 예측 결과와 정확도 정보를 템플릿에 전달하여 렌더링
    return render_template('predict.html', prediction=prediction, accuracy=accuracy)
