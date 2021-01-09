from flask import Flask, render_template, Response
from Base import open_webcam, end_recording

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/prediction', methods=['GET'])
def prediction():
    return render_template('Predicting.html')


@app.route('/end_prediction', methods=['GET'])
def end_prediction():
    return Response(end_recording(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed')
def video_feed():
    return Response(open_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
