from flask import Flask, g, render_template, request
import os
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'some super secret key!'
MAX_BUFFER_SIZE = 50 * 1000 * 1000  # 50 MB
socketio = SocketIO(app, logger=True, max_http_buffer_size=MAX_BUFFER_SIZE)
