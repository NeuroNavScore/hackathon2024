import sys
import socket
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget

class ServerThread(QThread):
    message_received = Signal(str)

    def __init__(self, host='localhost', port=12345):
        super().__init__()
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)

    def run(self):
        while True:
            client_socket, addr = self.server_socket.accept()
            data = client_socket.recv(1024).decode('utf-8')
            self.message_received.emit(data)
            client_socket.close()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Server App")
        self.setGeometry(100, 100, 600, 400)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.server_thread = ServerThread()
        self.server_thread.message_received.connect(self.display_message)
        self.server_thread.start()

    def display_message(self, message):
        self.text_edit.append(message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())