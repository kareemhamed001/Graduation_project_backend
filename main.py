from app import socketio, app
import controllers.routes
import controllers.sockets

if __name__ == '__main__':
    controllers.routes.init()
    controllers.sockets.init()
    socketio.run(app, port=8000, debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
