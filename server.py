import sys
import socket
import threading
import time
import json
import math
import random
from PySide6.QtCore import QThread, Signal, QObject
from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QLabel
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np

class ClientHandler(QThread):
    message_sent = Signal(str)

    def __init__(self, client_socket, address, model=None):
        super().__init__()
        self.client_socket = client_socket
        self.address = address
        self.is_running = True
        self.model = model  # Optional ML model

    def run(self):
        print(f"Client connected from {self.address}")
        start_time = time.time()
        while self.is_running:
            elapsed = time.time() - start_time

            # Simulate EEG data as sinusoidal waves with noise
            eeg = [math.sin(2 * math.pi * 0.5 * elapsed + i) * 50 + random.uniform(-10, 10) for i in range(8)]
            
            # Optional: Predict navigation direction using ML model
            if self.model:
                eeg_array = np.array(eeg).reshape(1, -1)
                prediction = self.model.predict(eeg_array)[0]
            else:
                prediction = random.choice(['N', 'S', 'E', 'W'])

            data = {
                'timestamp': elapsed,
                'eeg': eeg,
                'navigation': {
                    'position': [random.randint(0, 100), random.randint(0, 100)],
                    'direction': prediction
                }
            }
            message = json.dumps(data)

            # Send EEG data to the client
            try:
                self.client_socket.sendall(message.encode('utf-8') + b'\n')
                self.message_sent.emit(message)
            except BrokenPipeError:
                print(f"Client {self.address} disconnected")
                break
# ---------------------------------------------------------------------------------- 
            # NEW: Check for incoming data (e.g., triggers from Unity)
            try:
                incoming_data = self.client_socket.recv(1024).decode('utf-8').strip()
                if incoming_data:
                    print(f"[Server] Received trigger: {incoming_data}")
                    self.message_sent.emit(f"[Server] Received trigger: {incoming_data}")
                    #draw line in data
                    try:
                        # Parse as JSON if applicable
                        json_data = json.loads(incoming_data)
                        print(f"[Server] Parsed trigger JSON: {json_data}")
                        self.message_sent.emit(f"[Server] Parsed trigger JSON: {json_data}")
                    except json.JSONDecodeError:
                        print(f"[Server] Invalid JSON received: {incoming_data}")
                        self.message_sent.emit(f"[Server] Invalid JSON received: {incoming_data}")
            except socket.timeout:
                # Continue if no data is received within the timeout
                pass
            except ConnectionResetError:
                print(f"[Server] Connection reset by Unity client at {self.address}")
                break

# -----------------------------------------------------------------------------------
            time.sleep(1)  # Send data every second

    def stop(self):
        self.is_running = False
        self.quit()
        self.wait()
        self.client_socket.close()

class ServerThread(QThread):
    message_sent = Signal(str)

    def __init__(self, host='0.0.0.0', port=12345, model=None):
        super().__init__()
        self.host = host
        self.port = port
        self.is_running = True
        self.model = model
        self.client_handlers = []

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((self.host, self.port))
            server_socket.listen(5)
            server_socket.settimeout(1.0)  # periodic checks for self.is_running
            print(f"Server listening on {self.host}:{self.port}")
            while self.is_running:
                try:
                    client_socket, addr = server_socket.accept()
                    print(f"[ServerThread] Client connected: {addr}")  # Debugging log
                    handler = ClientHandler(client_socket, addr, self.model)
                    handler.message_sent.connect(self.message_sent.emit)
                    handler.start()
                    self.client_handlers.append(handler)
                except socket.timeout:
                    continue
        # Cleanup
        for handler in self.client_handlers:
            handler.stop()

    def stop(self):
        self.is_running = False
        self.quit()
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self, server_host='0.0.0.0', server_port=12345):
        super().__init__()
        self.setWindowTitle("Server Application")
        self.setGeometry(100, 100, 700, 500)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)

        self.status_label = QLabel(f"Server running on {server_host}:{server_port}")

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.text_edit)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Load ML model if available
        try:
            with open('predictor_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print("ML model loaded successfully.")
        except FileNotFoundError:
            print("ML model not found. Continuing without it.")
            self.model = None

        self.server_thread = ServerThread(host=server_host, port=server_port, model=self.model)
        self.server_thread.message_sent.connect(self.display_message)
        self.server_thread.start()

    def display_message(self, message):
        # Pretty-print JSON data
        try:
            data = json.loads(message)
            pretty = json.dumps(data, indent=4)
            self.text_edit.append(pretty)
        except json.JSONDecodeError:
            self.text_edit.append(message)

    def closeEvent(self, event):
        self.server_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    server_ip = '0.0.0.0'  # Listen on all interfaces
    server_port = 12345
    window = MainWindow(server_host=server_ip, server_port=server_port)
    window.show()
    sys.exit(app.exec())