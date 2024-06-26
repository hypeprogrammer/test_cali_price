from flask import Flask, render_template
from package.visual import visual_bp
from package.predict import predict_bp

app = Flask(__name__)

app.register_blueprint(visual_bp, url_prefix='/visual')
app.register_blueprint(predict_bp, url_prefix='/predict')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
