from flask import Flask, render_template
from visual import visual_blueprint
from predict import predict_blueprint

app = Flask(__name__)

app.register_blueprint(visual_blueprint, url_prefix='/visual')
app.register_blueprint(predict_blueprint, url_prefix='/predict')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
