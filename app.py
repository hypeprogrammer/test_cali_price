from flask import Flask, request, jsonify, send_file, render_template
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

app = Flask(__name__)

# 캘리포니아 주택가격 데이터 로드 및 모델 훈련
def load_data_and_train_model():
    california_housing = fetch_california_housing()
    data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    data['MedHouseVal'] = california_housing.target

    X = data.drop('MedHouseVal', axis=1)
    y = data['MedHouseVal']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test

model, X_test, y_test = load_data_and_train_model()

# 시각화 함수들
def load_data():
    california_housing = fetch_california_housing()
    data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    data['MedHouseVal'] = california_housing.target
    return data

def plot_scatter(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='MedInc', y='MedHouseVal', data=data)
    plt.title('Scatter plot of Median Income vs Median House Value')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

def plot_heatmap(data):
    plt.figure(figsize=(10, 6))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('Heatmap of Pearson Correlation Coefficients')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

def plot_histogram(data):
    plt.figure(figsize=(10, 6))
    data['MedHouseVal'].plot(kind='hist', bins=30, color='skyblue')
    plt.title('Histogram of Median House Value')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

def plot_boxplot(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='MedInc', y='MedHouseVal', data=data)
    plt.title('Boxplot of Median House Value by Median Income')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

def plot_piechart(data):
    plt.figure(figsize=(10, 6))
    data['MedHouseValCat'] = pd.qcut(data['MedHouseVal'], 5, labels=False)
    data['MedHouseValCat'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Pie chart of House Value Categories')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

@app.route('/visual/scatter', methods=['GET'])
def visual_scatter():
    data = load_data()
    img = plot_scatter(data)
    return send_file(img, mimetype='image/png')

@app.route('/visual/heatmap', methods=['GET'])
def visual_heatmap():
    data = load_data()
    img = plot_heatmap(data)
    return send_file(img, mimetype='image/png')

@app.route('/visual/histogram', methods=['GET'])
def visual_histogram():
    data = load_data()
    img = plot_histogram(data)
    return send_file(img, mimetype='image/png')

@app.route('/visual/boxplot', methods=['GET'])
def visual_boxplot():
    data = load_data()
    img = plot_boxplot(data)
    return send_file(img, mimetype='image/png')

@app.route('/visual/piechart', methods=['GET'])
def visual_piechart():
    data = load_data()
    img = plot_piechart(data)
    return send_file(img, mimetype='image/png')

# 예측을 수행하고 결과를 반환하는 함수
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.form  # HTML 폼으로부터 입력 데이터를 받음
    input_df = pd.DataFrame([input_data.to_dict(flat=True)])

    # 예측 수행
    prediction = model.predict(input_df)

    # 결과를 HTML 페이지로 반환
    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)