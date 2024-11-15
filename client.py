import sys
import socket
import json
import csv
import numpy as np  # Required for image handling with pyqtgraph
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QTextEdit, QLabel, QHBoxLayout, QLineEdit, QMessageBox, QFormLayout,
    QGroupBox, QFileDialog, QProgressBar, QSpinBox
)
from PySide6.QtCore import QThread, Signal, Slot, QObject, QTimer, Qt
import pyqtgraph as pg
from collections import deque
import random
import time

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
            print(f"[ReceiverThread] Connected to {self.host}:{self.port}")
        except ConnectionRefusedError:
            self.connection_status.emit("Failed to connect to the server.")
            print("[ReceiverThread] Failed to connect to the server.")
            return

        buffer = ""
        while self.is_running:
            try:
                data = self.sock.recv(1024).decode('utf-8')
                if not data:
                    self.connection_status.emit("Server disconnected.")
                    print("[ReceiverThread] Server disconnected.")
                    break
                buffer += data
                while '\n' in buffer:
                    message, buffer = buffer.split('\n', 1)
                    try:
                        json_data = json.loads(message)
                        self.data_received.emit(json_data)
                        print(f"[ReceiverThread] Data received: {json_data}")
                    except json.JSONDecodeError:
                        print("Received malformed JSON data.")
            except ConnectionResetError:
                self.connection_status.emit("Connection lost.")
                print("[ReceiverThread] Connection lost.")
                break

    def stop(self):
        print("[ReceiverThread] Stopping thread.")
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
        print("[ClientWindow] Initializing ClientWindow.")
        self.setWindowTitle("NeuroNavScore Client Application")
        self.setGeometry(100, 100, 800, 600)  # Reduced size for simplicity

        # Initialize Test Control Variables BEFORE init_ui()
        self.test_running = False
        self.test_paused = False
        self.test_start_time = None
        self.test_duration = 10  # default duration in seconds
        self.elapsed_time = 0
        self.pass_fail_result = None
        self.score = 0

        # Initialize Signal Handler
        self.signal_handler = SignalHandler()
        self.signal_handler.data_received.connect(self.process_data)
        self.signal_handler.connection_status.connect(self.update_status)

        # Initialize UI and Socket AFTER setting test variables
        self.init_ui()
        self.init_socket(server_ip, server_port)

        # Timer for Test Monitoring
        self.test_timer = QTimer()
        self.test_timer.timeout.connect(self.monitor_test)

    def init_ui(self):
        print("[ClientWindow] Setting up UI.")
        # Main Layout
        main_layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("NeuroNavScore")
        title_label.setStyleSheet("font-size: 32px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)

        # Patient Information Group
        patient_group = QGroupBox("Patient Information")
        patient_layout = QFormLayout()

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter patient's full name")
        self.age_input = QLineEdit()
        self.age_input.setPlaceholderText("Enter patient's age")

        patient_layout.addRow(QLabel("Patient Name:"), self.name_input)
        patient_layout.addRow(QLabel("Age:"), self.age_input)
        patient_group.setLayout(patient_layout)

        # Connection Status
        self.status_label = QLabel("Connection Status: Disconnected")
        self.status_label.setStyleSheet("font-weight: bold; color: red;")
        self.status_label.setAlignment(Qt.AlignCenter)

        # EEG Data Visualization
        eeg_group = QGroupBox("EEG Data")
        eeg_layout = QVBoxLayout()
        self.eeg_graph = pg.PlotWidget(title="Real-Time EEG Data")
        self.eeg_graph.setYRange(-150, 150)
        self.eeg_graph.showGrid(x=True, y=True)
        self.eeg_graph.addLegend()

        self.eeg_channels = 8
        self.curves = []
        self.colors = ['r', 'g', 'b', 'c', 'm', 'y', 'w', 'k']

        for i in range(self.eeg_channels):
            curve = self.eeg_graph.plot(pen=pg.mkPen(self.colors[i], width=1), name=f"Channel {i+1}")
            self.curves.append(curve)

        eeg_layout.addWidget(self.eeg_graph)
        eeg_group.setLayout(eeg_layout)

        # Test Controls Group
        controls_group = QGroupBox("Test Controls")
        controls_layout = QHBoxLayout()

        # Duration Setting
        self.duration_label = QLabel("Test Duration (s):")
        self.duration_input = QSpinBox()
        self.duration_input.setRange(5, 300)  # Allow durations between 5 and 300 seconds
        self.duration_input.setValue(self.test_duration)  # Set initial value
        self.duration_input.valueChanged.connect(self.update_test_duration)

        # Buttons
        self.start_test_button = QPushButton("Start Test")
        self.start_test_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.start_test_button.clicked.connect(self.start_test)

        self.pause_test_button = QPushButton("Pause Test")
        self.pause_test_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.pause_test_button.clicked.connect(self.pause_test)
        self.pause_test_button.setEnabled(False)

        self.stop_test_button = QPushButton("Stop Test")
        self.stop_test_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.stop_test_button.clicked.connect(self.stop_test)
        self.stop_test_button.setEnabled(False)

        self.reset_test_button = QPushButton("Reset Test")
        self.reset_test_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.reset_test_button.clicked.connect(self.reset_test)
        self.reset_test_button.setEnabled(False)

        controls_layout.addWidget(self.duration_label)
        controls_layout.addWidget(self.duration_input)
        controls_layout.addWidget(self.start_test_button)
        controls_layout.addWidget(self.pause_test_button)
        controls_layout.addWidget(self.stop_test_button)
        controls_layout.addWidget(self.reset_test_button)
        controls_group.setLayout(controls_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.test_duration)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Test Progress: %p%")
        self.progress_bar.setStyleSheet("QProgressBar::chunk {background-color: #05B8CC;}")

        # Performance Indicators Group
        performance_group = QGroupBox("Performance Indicators")
        performance_layout = QHBoxLayout()

        self.score_label = QLabel("Visuospatial Processing Score: 0")
        self.score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: blue;")
        self.score_label.setAlignment(Qt.AlignCenter)

        performance_layout.addWidget(self.score_label)
        performance_group.setLayout(performance_layout)

        # Export Results Button
        self.export_button = QPushButton("Export Results")
        self.export_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)

        # Assemble Main Layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(patient_group)
        main_layout.addWidget(self.status_label)
        main_layout.addWidget(eeg_group)
        main_layout.addWidget(controls_group)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(performance_group)
        main_layout.addWidget(self.export_button)

        # Set Main Layout
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def update_test_duration(self, value):
        print(f"[ClientWindow] Test duration updated to {value} seconds.")
        self.test_duration = value
        if not self.test_running:
            self.progress_bar.setMaximum(self.test_duration)
            self.progress_bar.setFormat(f"Test Progress: 0/{self.test_duration} seconds")

    def draw_grid(self):
        # Navigation grid has been removed as per requirements
        pass

    def init_socket(self, host, port):
        print(f"[ClientWindow] Initializing socket connection to {host}:{port}.")
        self.receiver_thread = ReceiverThread(host, port)
        self.receiver_thread.data_received.connect(self.signal_handler.data_received)
        self.receiver_thread.connection_status.connect(self.signal_handler.connection_status)
        self.receiver_thread.start()

    @Slot(str)
    def update_status(self, status):
        print(f"[ClientWindow] Connection status updated: {status}")
        self.status_label.setText(f"Connection Status: {status}")
        if status.startswith("Connected"):
            self.status_label.setStyleSheet("font-weight: bold; color: green;")
        else:
            self.status_label.setStyleSheet("font-weight: bold; color: red;")

    @Slot(dict)
    def process_data(self, data):
        print(f"[ClientWindow] Processing data: {data}")
        eeg = data.get('eeg', [])
        timestamp = data.get('timestamp', 0)

        # Update EEG data
        for i in range(min(self.eeg_channels, len(eeg))):
            self.eeg_data[i].append(eeg[i])

        # Update EEG Graph
        for i, curve in enumerate(self.curves):
            curve.setData(list(self.eeg_data[i]))

        # Update Score (Dummy Calculation based on EEG data sum)
        if self.test_running and not self.test_paused:
            # Example: Sum of absolute EEG values as a simple score
            eeg_sum = sum(abs(val) for val in eeg[:self.eeg_channels])
            self.score = min(eeg_sum, 100)  # Clamp score to 100
            self.score_label.setText(f"Visuospatial Processing Score: {self.score}")
            print(f"[ClientWindow] Updated Score: {self.score}")

    def start_test(self):
        print("[ClientWindow] Start Test button clicked.")
        if self.test_running:
            QMessageBox.warning(self, "Test Running", "A test is already in progress.")
            return

        name = self.name_input.text().strip()
        age = self.age_input.text().strip()

        if not name or not age:
            QMessageBox.warning(self, "Input Error", "Please enter both patient name and age.")
            return

        if not age.isdigit():
            QMessageBox.warning(self, "Input Error", "Age must be a number.")
            return

        # Update test_duration based on QSpinBox value
        self.test_duration = self.duration_input.value()
        self.progress_bar.setMaximum(self.test_duration)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat(f"Test Progress: 0/{self.test_duration} seconds")
        print(f"[ClientWindow] Test duration set to {self.test_duration} seconds.")

        # Initialize Test Variables
        self.test_running = True
        self.test_paused = False
        self.test_start_time = time.time()
        self.elapsed_time = 0
        self.pass_fail_result = None
        self.score = 0
        self.score_label.setText("Visuospatial Processing Score: 0")
        self.eeg_graph.clear()  # Clear existing EEG plots
        for i in range(self.eeg_channels):
            self.curves[i] = self.eeg_graph.plot(pen=pg.mkPen(self.colors[i], width=1), name=f"Channel {i+1}")

        self.start_test_button.setEnabled(False)
        self.pause_test_button.setEnabled(True)
        self.stop_test_button.setEnabled(True)
        self.reset_test_button.setEnabled(False)
        self.export_button.setEnabled(False)

        # Start Timer for Progress Bar
        self.test_timer.start(1000)  # Update every second
        print("[ClientWindow] Test timer started.")

        # Inform Clinician
        QMessageBox.information(self, "Test Started", "Navigation test has begun.")

    def pause_test(self):
        print("[ClientWindow] Pause Test button clicked.")
        if not self.test_running:
            return

        if not self.test_paused:
            self.test_paused = True
            self.test_timer.stop()
            self.pause_test_button.setText("Resume Test")
            QMessageBox.information(self, "Test Paused", "Navigation test has been paused.")
            print("[ClientWindow] Test paused.")
        else:
            self.test_paused = False
            self.test_timer.start(1000)
            self.pause_test_button.setText("Pause Test")
            QMessageBox.information(self, "Test Resumed", "Navigation test has resumed.")
            print("[ClientWindow] Test resumed.")

    def stop_test(self):
        print("[ClientWindow] Stop Test button clicked.")
        if not self.test_running:
            return

        self.test_timer.stop()
        self.test_running = False
        self.test_paused = False
        self.start_test_button.setEnabled(True)
        self.pause_test_button.setEnabled(False)
        self.pause_test_button.setText("Pause Test")
        self.stop_test_button.setEnabled(False)
        self.reset_test_button.setEnabled(True)

        # Determine Pass/Fail based on score
        if self.score >= 50:  # Example threshold
            self.pass_fail_result = 'Pass'
            color = "green"
        else:
            self.pass_fail_result = 'Fail'
            color = "red"

        # Update Score Label to Highlight Pass/Fail
        self.score_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
        self.score_label.setText(f"Visuospatial Processing Score: {self.score} - {self.pass_fail_result}")

        # Enable Export Button
        if self.pass_fail_result in ['Pass', 'Fail']:
            self.export_button.setEnabled(True)

        # Inform Clinician
        QMessageBox.information(self, "Test Stopped", f"Navigation test stopped with result: {self.pass_fail_result}")
        print(f"[ClientWindow] Test stopped with result: {self.pass_fail_result}")

    def reset_test(self):
        print("[ClientWindow] Reset Test button clicked.")
        if self.test_running:
            return

        self.pass_fail_result = None
        self.score = 0
        self.score_label.setText("Visuospatial Processing Score: 0")
        self.score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: blue;")
        self.progress_bar.setValue(0)
        self.nav_text.clear()
        self.eeg_graph.clear()
        for i in range(self.eeg_channels):
            self.curves[i] = self.eeg_graph.plot(pen=pg.mkPen(self.colors[i], width=1), name=f"Channel {i+1}")

        self.reset_test_button.setEnabled(False)
        print("[ClientWindow] All test data has been reset.")
        QMessageBox.information(self, "Test Reset", "All test data has been reset.")

    def export_results(self):
        print("[ClientWindow] Export Results button clicked.")
        if not self.pass_fail_result:
            QMessageBox.warning(self, "No Results", "There are no results to export.")
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=options
        )
        if file_path:
            try:
                with open(file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        "Patient Name",
                        "Age",
                        "Result",
                        "Visuospatial Processing Score"
                    ])
                    writer.writerow([
                        self.name_input.text().strip(),
                        self.age_input.text().strip(),
                        self.pass_fail_result,
                        self.score
                    ])
                QMessageBox.information(self, "Export Successful", f"Results exported to {file_path}")
                print(f"[ClientWindow] Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"An error occurred while exporting results:\n{e}")
                print(f"[ClientWindow] Export failed: {e}")

    def monitor_test(self):
        self.elapsed_time += 1
        self.progress_bar.setValue(self.elapsed_time)
        print(f"[ClientWindow] Test progress: {self.elapsed_time}/{self.test_duration} seconds.")

        if self.elapsed_time >= self.test_duration:
            print("[ClientWindow] Test duration reached. Ending test.")
            self.end_test()

    def end_test(self):
        print("[ClientWindow] Ending test due to duration.")
        self.test_timer.stop()
        self.test_running = False
        self.test_paused = False
        self.start_test_button.setEnabled(True)
        self.pause_test_button.setEnabled(False)
        self.pause_test_button.setText("Pause Test")
        self.stop_test_button.setEnabled(False)
        self.reset_test_button.setEnabled(True)

        # Determine Pass/Fail based on score
        if self.score >= 50:  # Example threshold
            self.pass_fail_result = 'Pass'
            color = "green"
        else:
            self.pass_fail_result = 'Fail'
            color = "red"

        # Update Score Label to Highlight Pass/Fail
        self.score_label.setStyleSheet(f"font-size: 24px; font-weight: bold; color: {color};")
        self.score_label.setText(f"Visuospatial Processing Score: {self.score} - {self.pass_fail_result}")

        # Enable Export Button
        if self.pass_fail_result in ['Pass', 'Fail']:
            self.export_button.setEnabled(True)

        # Inform Clinician
        QMessageBox.information(self, "Test Completed", f"Navigation test completed with result: {self.pass_fail_result}")
        print(f"[ClientWindow] Test completed with result: {self.pass_fail_result}")

    def closeEvent(self, event):
        print("[ClientWindow] Closing application.")
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
