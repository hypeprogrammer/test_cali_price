from flask import Blueprint, send_file, make_response, render_template  # Flask 블루프린트, 파일 전송, 응답 생성, 템플릿 렌더링을 위한 모듈 임포트
import pandas as pd  # 데이터 처리를 위한 라이브러리 임포트
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리 임포트
import seaborn as sns  # 데이터 시각화를 위한 라이브러리 임포트
import io  # 입출력 처리를 위한 모듈 임포트
from sklearn.datasets import fetch_california_housing  # 캘리포니아 주택가격 데이터셋 로드를 위한 모듈 임포트
import warnings  # 경고 메시지 처리를 위한 모듈 임포트

# 코드 경고 무시
warnings.filterwarnings('ignore')

# 한글 글꼴 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 한글 글꼴에서 마이너스 기호 깨지지 않도록 설정

# 블루프린트 생성
visual_bp = Blueprint('visual_bp', __name__)

# 캘리포니아 주택가격 데이터 로드
housing_data = fetch_california_housing()
housing = pd.DataFrame(housing_data.data, columns=housing_data.feature_names)  # 데이터프레임으로 변환
housing['MedHouseVal'] = housing_data.target  # 타겟 값 추가

def load_data():
    # 데이터 로드 함수
    return housing

def plot_scatter(data):
    # 산점도 그리기 함수
    plt.figure(figsize=(10, 6))  # 그래프 크기 설정
    sns.scatterplot(x='MedInc', y='MedHouseVal', data=data)  # 산점도 그리기
    plt.title('Scatter plot of Median Income vs Median House Value')  # 그래프 제목 설정
    img = io.BytesIO()  # 이미지 저장을 위한 바이너리 스트림 생성
    plt.savefig(img, format='png')  # 이미지를 PNG 형식으로 저장
    img.seek(0)  # 스트림의 시작 위치로 이동
    plt.close()  # 그래프 닫기
    return img  # 이미지 반환

def plot_heatmap(data):
    # 히트맵 그리기 함수
    plt.figure(figsize=(10, 6))  # 그래프 크기 설정
    corr = data.corr()  # 상관 행렬 계산
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)  # 히트맵 그리기
    plt.title('Heatmap of Pearson Correlation Coefficients')  # 그래프 제목 설정
    img = io.BytesIO()  # 이미지 저장을 위한 바이너리 스트림 생성
    plt.savefig(img, format='png')  # 이미지를 PNG 형식으로 저장
    img.seek(0)  # 스트림의 시작 위치로 이동
    plt.close()  # 그래프 닫기
    return img  # 이미지 반환

def plot_histogram(data):
    # 히스토그램 그리기 함수
    plt.figure(figsize=(10, 6))  # 그래프 크기 설정
    data['MedHouseVal'].plot(kind='hist', bins=30, color='skyblue')  # 히스토그램 그리기
    plt.title('Histogram of Median House Value')  # 그래프 제목 설정
    img = io.BytesIO()  # 이미지 저장을 위한 바이너리 스트림 생성
    plt.savefig(img, format='png')  # 이미지를 PNG 형식으로 저장
    img.seek(0)  # 스트림의 시작 위치로 이동
    plt.close()  # 그래프 닫기
    return img  # 이미지 반환

def plot_boxplot(data):
    # 박스플롯 그리기 함수
    plt.figure(figsize=(10, 6))  # 그래프 크기 설정
    sns.boxplot(x='MedInc', y='MedHouseVal', data=data)  # 박스플롯 그리기
    plt.title('Boxplot of Median House Value by Median Income')  # 그래프 제목 설정
    img = io.BytesIO()  # 이미지 저장을 위한 바이너리 스트림 생성
    plt.savefig(img, format='png')  # 이미지를 PNG 형식으로 저장
    img.seek(0)  # 스트림의 시작 위치로 이동
    plt.close()  # 그래프 닫기
    return img  # 이미지 반환

def plot_piechart(data):
    # 파이차트 그리기 함수
    plt.figure(figsize=(10, 6))  # 그래프 크기 설정
    data['MedHouseValCat'] = pd.qcut(data['MedHouseVal'], 5, labels=False)  # 타겟 값을 5개의 카테고리로 나누기
    data['MedHouseValCat'].value_counts().plot(kind='pie', autopct='%1.1f%%')  # 파이차트 그리기
    plt.title('Pie chart of House Value Categories')  # 그래프 제목 설정
    img = io.BytesIO()  # 이미지 저장을 위한 바이너리 스트림 생성
    plt.savefig(img, format='png')  # 이미지를 PNG 형식으로 저장
    img.seek(0)  # 스트림의 시작 위치로 이동
    plt.close()  # 그래프 닫기
    return img  # 이미지 반환

@visual_bp.route('/scatter', methods=['GET'])
def visual_scatter():
    # 산점도 시각화 라우트
    data = load_data()  # 데이터 로드
    img = plot_scatter(data)  # 산점도 생성
    return send_file(img, mimetype='image/png')  # 이미지 파일 전송

@visual_bp.route('/heatmap', methods=['GET'])
def visual_heatmap():
    # 히트맵 시각화 라우트
    data = load_data()  # 데이터 로드
    img = plot_heatmap(data)  # 히트맵 생성
    return send_file(img, mimetype='image/png')  # 이미지 파일 전송

@visual_bp.route('/histogram', methods=['GET'])
def visual_histogram():
    # 히스토그램 시각화 라우트
    data = load_data()  # 데이터 로드
    img = plot_histogram(data)  # 히스토그램 생성
    return send_file(img, mimetype='image/png')  # 이미지 파일 전송

@visual_bp.route('/boxplot', methods=['GET'])
def visual_boxplot():
    # 박스플롯 시각화 라우트
    data = load_data()  # 데이터 로드
    img = plot_boxplot(data)  # 박스플롯 생성
    return send_file(img, mimetype='image/png')  # 이미지 파일 전송

@visual_bp.route('/piechart', methods=['GET'])
def visual_piechart():
    # 파이차트 시각화 라우트
    data = load_data()  # 데이터 로드
    img = plot_piechart(data)  # 파이차트 생성
    return send_file(img, mimetype='image/png')  # 이미지 파일 전송

@visual_bp.route('/scatter_location')
def scatter_plot():
    # 위도와 경도에 따른 산점도 시각화 라우트
    img = io.BytesIO()  # 이미지 저장을 위한 바이너리 스트림 생성

    # 산점도 생성
    housing.plot(kind='scatter', x='Longitude', y='Latitude', figsize=(8, 7), alpha=0.1)  # 산점도 그리기
    plt.xlabel('경도')  # x축 레이블 설정
    plt.ylabel('위도')  # y축 레이블 설정
    plt.title('위·경도에 따른 산점도 (밀집지역)')  # 그래프 제목 설정

    # 이미지를 BytesIO 객체에 저장
    plt.savefig(img, format='png')  # 이미지를 PNG 형식으로 저장
    img.seek(0)  # 스트림의 시작 위치로 이동
    plt.close()  # 그래프 닫기

    # 이미지를 base64로 인코딩하여 반환
    response = make_response(send_file(img, mimetype='image/png'))  # 이미지 파일 전송
    response.headers['Content-Disposition'] = 'inline; filename=scatter.png'  # 파일 이름 설정
    return response  # 응답 반환

@visual_bp.route('/pairplot')
def pairplot():
    # 페어플롯 시각화 라우트
    img = io.BytesIO()  # 이미지 저장을 위한 바이너리 스트림 생성

    # 페어플롯 생성
    sns.pairplot(housing, corner=True)  # 페어플롯 그리기
    plt.savefig(img, format='png')  # 이미지를 PNG 형식으로 저장
    img.seek(0)  # 스트림의 시작 위치로 이동
    plt.close()  # 그래프 닫기

    # 이미지를 base64로 인코딩하여 반환
    response = make_response(send_file(img, mimetype='image/png'))  # 이미지 파일 전송
    response.headers['Content-Disposition'] = 'inline; filename=pairplot.png'  # 파일 이름 설정
    return response  # 응답 반환

@visual_bp.route('/')
def visual_index():
    # 시각화 인덱스 페이지 라우트
    return render_template('visual.html')  # 인덱스 페이지 렌더링
