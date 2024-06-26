from flask import Flask, render_template  # Flask 애플리케이션과 템플릿 렌더링을 위한 모듈 임포트
from package.visual import visual_bp  # 시각화 블루프린트를 패키지에서 임포트
from package.predict import predict_bp  # 예측 블루프린트를 패키지에서 임포트

app = Flask(__name__)  # Flask 애플리케이션 인스턴스 생성

# 시각화 블루프린트를 '/visual' 경로로 등록
app.register_blueprint(visual_bp, url_prefix='/visual')
# 예측 블루프린트를 '/predict' 경로로 등록
app.register_blueprint(predict_bp, url_prefix='/predict')

@app.route('/')
def index():
    # 루트 경로('/')에 대한 요청을 처리하는 함수
    return render_template('index.html')  # 'index.html' 템플릿을 렌더링하여 반환

if __name__ == '__main__':
    # 이 파일이 직접 실행될 때만 Flask 애플리케이션을 실행
    app.run(debug=True)  # 디버그 모드로 애플리케이션 실행
