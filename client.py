import sys
import socket
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QTextEdit, QLineEdit

class Client(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()
        self.initSocket()

    def initUI(self):
        self.setWindowTitle("Client")

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)

        self.input_line = QLineEdit(self)
        self.input_line.setPlaceholderText("Enter message")

        self.send_button = QPushButton("Send", self)
        self.send_button.clicked.connect(self.send_message)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.input_line)
        layout.addWidget(self.send_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def initSocket(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(('localhost', 12345))

    def send_message(self):
        message = self.input_line.text()
        self.client_socket.sendall(message.encode('utf-8'))
        self.text_edit.append(f"Sent: {message}")
        self.input_line.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    client = Client()
    client.show()
    sys.exit(app.exec())