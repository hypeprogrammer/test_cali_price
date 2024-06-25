from flask import Blueprint, request, render_template
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

predict_blueprint = Blueprint('predict', __name__)

# 캘리포니아 주택가격 데이터 로드 및 모델 훈련
def load_data_and_train_model():
    california_housing = fetch_california_housing()
    data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    data['MedHouseVal'] = california_housing.target

    X = data.drop('MedHouseVal', axis=1)
    y = data['MedHouseVal']

    # 로그 변환을 적용
    y = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 스케일링 적용
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test, scaler

model, X_test, y_test, scaler = load_data_and_train_model()

@predict_blueprint.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        input_data = request.form
        input_df = pd.DataFrame([input_data.to_dict(flat=True)])
        input_df = input_df.astype(float)

        # 스케일링 적용
        input_df = scaler.transform(input_df)

        # 예측 수행
        prediction = model.predict(input_df)

        # 로그 변환 역변환
        prediction = np.expm1(prediction)

        return render_template('predict.html', prediction=prediction[0])
    return render_template('predict.html')
