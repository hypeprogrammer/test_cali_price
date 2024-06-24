from flask import Flask, send_file
from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

app = Flask(__name__)

# 캘리포니아 주택가격 데이터 로드
def load_data():
    california_housing = fetch_california_housing()
    data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    data['MedHouseVal'] = california_housing.target
    return data

# 산포도 시각화 함수
def plot_scatter(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='MedInc', y='MedHouseVal', data=data)
    plt.title('Scatter plot of Median Income vs Median House Value')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

# 피어슨 상관계수 히트맵 시각화 함수
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

# 히스토그램 시각화 함수
def plot_histogram(data):
    plt.figure(figsize=(10, 6))
    data['MedHouseVal'].plot(kind='hist', bins=30, color='skyblue')
    plt.title('Histogram of Median House Value')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

# 박스플롯 시각화 함수
def plot_boxplot(data):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='ocean_proximity', y='MedHouseVal', data=data)
    plt.title('Boxplot of Median House Value by Ocean Proximity')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return img

# 파이차트 시각화 함수
def plot_piechart(data):
    plt.figure(figsize=(10, 6))
    data['ocean_proximity'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Pie chart of Ocean Proximity')
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

if __name__ == '__main__':
    app.run(debug=True)
