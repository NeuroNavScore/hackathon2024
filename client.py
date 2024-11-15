import sys
import socket
import json
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QTextEdit, QLabel, QHBoxLayout
)
from PySide6.QtCore import QThread, Signal, Slot, QObject
import pyqtgraph as pg
from collections import deque

class ReceiverThread(QThread):
    data_received = Signal(dict)
    connection_status = Signal(str)

    def __init__(self, host='localhost', port=12345):
        super().__init__()
        self.host = host
        self.port = port
        self.is_running = True
        self.sock = None

    def run(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            self.connection_status.emit(f"Connected to {self.host}:{self.port}")
        except ConnectionRefusedError:
            self.connection_status.emit("Failed to connect to the server.")
            return

        buffer = ""
        while self.is_running:
            try:
                data = self.sock.recv(1024).decode('utf-8')
                if not data:
                    self.connection_status.emit("Server disconnected.")
                    break
                buffer += data
                while '\n' in buffer:
                    message, buffer = buffer.split('\n', 1)
                    try:
                        json_data = json.loads(message)
                        self.data_received.emit(json_data)
                    except json.JSONDecodeError:
                        print("Received malformed JSON data.")
            except ConnectionResetError:
                self.connection_status.emit("Connection lost.")
                break

    def stop(self):
        self.is_running = False
        if self.sock:
            self.sock.close()
        self.quit()
        self.wait()

class SignalHandler(QObject):
    data_received = Signal(dict)
    connection_status = Signal(str)

class ClientWindow(QMainWindow):
    def __init__(self, server_ip='localhost', server_port=12345):
        super().__init__()
        self.setWindowTitle("Client Application")
        self.setGeometry(800, 100, 1000, 600)

        # **Initialize data structures for EEG channels BEFORE calling init_ui**
        self.eeg_channels = 8
        self.eeg_data = [deque(maxlen=100) for _ in range(self.eeg_channels)]
        self.curves = []
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w', 'k']

        self.signal_handler = SignalHandler()
        self.signal_handler.data_received.connect(self.process_data)
        self.signal_handler.connection_status.connect(self.update_status)

        self.init_ui()
        self.init_socket(server_ip, server_port)

    def init_ui(self):
        # Status Label
        self.status_label = QLabel("Connection Status: Disconnected")

        # EEG Graph
        self.eeg_graph = pg.PlotWidget(title="EEG Data")
        self.eeg_graph.setYRange(-150, 150)
        self.eeg_graph.showGrid(x=True, y=True)
        self.eeg_graph.addLegend()
        for i in range(self.eeg_channels):
            curve = self.eeg_graph.plot(pen=pg.mkPen(self.colors[i], width=1), name=f"Channel {i+1}")
            self.curves.append(curve)

        # Navigation Status
        self.nav_label = QLabel("Navigation Status:")
        self.nav_text = QTextEdit()
        self.nav_text.setReadOnly(True)

        # Layout Setup
        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.eeg_graph)
        layout.addWidget(self.nav_label)
        layout.addWidget(self.nav_text)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def init_socket(self, host, port):
        self.receiver_thread = ReceiverThread(host, port)
        self.receiver_thread.data_received.connect(self.signal_handler.data_received)
        self.receiver_thread.connection_status.connect(self.signal_handler.connection_status)
        self.receiver_thread.start()

    @Slot(str)
    def update_status(self, status):
        self.status_label.setText(f"Connection Status: {status}")

    @Slot(dict)
    def process_data(self, data):
        eeg = data.get('eeg', [])
        navigation = data.get('navigation', {})
        timestamp = data.get('timestamp', 0)

        # Update EEG data
        for i in range(min(self.eeg_channels, len(eeg))):
            self.eeg_data[i].append(eeg[i])

        # Update Navigation Status
        nav_info = f"Timestamp: {timestamp:.2f}s\n"
        nav_info += f"Position: {navigation.get('position', [0, 0])}\n"
        nav_info += f"Direction: {navigation.get('direction', 'N/A')}\n"
        nav_info += "-"*30 + "\n"
        self.nav_text.append(nav_info)

        # Update EEG Graph
        for i, curve in enumerate(self.curves):
            curve.setData(list(self.eeg_data[i]))

    def closeEvent(self, event):
        if hasattr(self, 'receiver_thread'):
            self.receiver_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    server_ip = 'localhost'  # Change this to the server's IP address if on a different machine
    server_port = 12345
    client = ClientWindow(server_ip, server_port)
    client.show()
    sys.exit(app.exec())
