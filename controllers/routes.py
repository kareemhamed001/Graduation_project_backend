from flask import render_template, request, redirect, url_for
from app import app


@app.route('/', methods=['GET'])
def index():
    try:
        return render_template('index.html')
    except FileNotFoundError as exception:
        return render_template('index.html', exception_message='Video not found')


@app.route('/result', methods=['GET'])
def result():
    result = request.args.get('result')
    confidence = request.args.get('confidence')
    video = request.args.get('video')

    return render_template('result.html', result=result, confidence=confidence, video=video)


def init():
    print("Initializing api controller")
