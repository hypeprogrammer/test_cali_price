from flask import Blueprint, send_file, make_response, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.datasets import fetch_california_housing
import warnings

# 코드 경고 무시
warnings.filterwarnings('ignore')

# 한글 글꼴 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 한글 글꼴에서 마이너스 기호 깨지지 않도록

visual_blueprint = Blueprint('visual', __name__)

housing_data = fetch_california_housing()
housing = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)
housing['MedHouseVal'] = housing_data.target

def load_data():
    return housing

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

@visual_blueprint.route('/scatter', methods=['GET'])
def visual_scatter():
    data = load_data()
    img = plot_scatter(data)
    return send_file(img, mimetype='image/png')

@visual_blueprint.route('/heatmap', methods=['GET'])
def visual_heatmap():
    data = load_data()
    img = plot_heatmap(data)
    return send_file(img, mimetype='image/png')

@visual_blueprint.route('/histogram', methods=['GET'])
def visual_histogram():
    data = load_data()
    img = plot_histogram(data)
    return send_file(img, mimetype='image/png')

@visual_blueprint.route('/boxplot', methods=['GET'])
def visual_boxplot():
    data = load_data()
    img = plot_boxplot(data)
    return send_file(img, mimetype='image/png')

@visual_blueprint.route('/piechart', methods=['GET'])
def visual_piechart():
    data = load_data()
    img = plot_piechart(data)
    return send_file(img, mimetype='image/png')

@visual_blueprint.route('/scatter_location')
def scatter_plot():
    img = io.BytesIO()

    # 산점도 생성
    housing.plot(kind='scatter', x='Longitude', y='Latitude', figsize=(8, 7), alpha=0.1)
    plt.xlabel('경도')
    plt.ylabel('위도')
    plt.title('위·경도에 따른 산점도 (밀집지역)')

    # 이미지를 BytesIO 객체에 저장
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # 이미지를 base64로 인코딩하여 반환
    response = make_response(send_file(img, mimetype='image/png'))
    response.headers['Content-Disposition'] = 'inline; filename=scatter.png'
    return response

@visual_blueprint.route('/pairplot')
def pairplot():
    img = io.BytesIO()

    # 페어플롯 생성
    sns.pairplot(housing, corner=True)
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # 이미지를 base64로 인코딩하여 반환
    response = make_response(send_file(img, mimetype='image/png'))
    response.headers['Content-Disposition'] = 'inline; filename=pairplot.png'
    return response

@visual_blueprint.route('/')
def visual_index():
    return render_template('visual.html')
