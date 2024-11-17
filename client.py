import sys
import time
import json
import csv
import socket
from datetime import datetime
from collections import deque
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QTextEdit, QLabel, QHBoxLayout, QLineEdit, QMessageBox, QFormLayout,
    QGroupBox, QFileDialog, QProgressBar, QSpinBox
)
from PySide6.QtCore import QThread, Signal, Slot, QObject, QTimer, Qt
import pyqtgraph as pg

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow import DataFilter
from brainflow.data_filter import DetrendOperations, FilterTypes, NoiseTypes

TESTING = True
# Define the DataAcquisitionThread to handle BrainFlow data
class DataAcquisitionThread(QThread):
    eeg_data_signal = Signal(tuple)  # Emit EEG data list

    def __init__(self, board_id=BoardIds.SYNTHETIC_BOARD, params=None):
        super().__init__()
        self.board_id = board_id

        self.params = params if params else BrainFlowInputParams()
        self.board = None
        self.is_running = True
        self.eeg_channels = BoardShim.get_eeg_channels(board_id)

    def run(self):
        try:
            # Initialize the board
            self.board = BoardShim(self.board_id, self.params)
            self.board.prepare_session()
            self.board.start_stream()
            print("[DataAcquisitionThread] Board session started.")

            while self.is_running:
                # Get data from the board
                data = self.board.get_board_data()
                if data.size > 0:
                    # Assuming the first row contains the latest data for channel 1
                    # Adjust indexing based on your specific board and channel setup
                    latest_eeg = data[self.eeg_channels]
                    t = data[BoardShim.get_timestamp_channel(self.board_id)]
                    packet = (latest_eeg, t)
                    self.eeg_data_signal.emit(packet)

                    # Save data to a file
                    DataFilter.write_file(data, 'eeg_data.csv', 'a')
                time.sleep(0.05)  # Adjust the sleep time as needed

        except Exception as e:
            print(f"[DataAcquisitionThread] Exception: {e}")
            QMessageBox.critical(None, "Board Connection Error", f"An error occurred: {e}")
            self.is_running = False

    def stop(self):
        self.is_running = False
        if self.board and self.board.is_prepared():
            self.board.stop_stream()
            self.board.release_session()
            print("[DataAcquisitionThread] Board session released.")
        self.quit()
        self.wait()

# Define the MazeDataReceiverThread to handle incoming maze data
class MazeDataReceiverThread(QThread):
    maze_data_signal = Signal(dict)  # Emit maze data as a dictionary

    def __init__(self, host='0.0.0.0', port=12345):
        super().__init__()
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.is_running = True

    def run(self):
        if TESTING:
            while True:
                time.sleep(5)
                data = {"triggerID": "T3"}
                self.maze_data_signal.emit(data)
        try:
            # Set up the server socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)  # Timeout for accepting connections
            print(f"[MazeDataReceiverThread] Server listening on {self.host}:{self.port}")
    
            while self.is_running:
                try:
                    self.client_socket, addr = self.server_socket.accept()
                    print(f"[MazeDataReceiverThread] Connected by {addr}")
                    self.client_socket.settimeout(1.0)
                except socket.timeout:
                    continue  # Check if still running
                except OSError as e:
                    if not self.is_running:
                        break  # Socket was closed, exit the loop
                    else:
                        print(f"[MazeDataReceiverThread] OSError during accept: {e}")
                        continue

                while self.is_running:
                    try:
                        data = self.client_socket.recv(1024).decode('utf-8')
                        if not data:
                            print("[MazeDataReceiverThread] Client disconnected.")
                            break
                        
                        #draw line in data
                        try:
                            # Parse as JSON if applicable
                            json_data = json.loads(data)
                            print(f"[MazeDataReceiverThread] Parsed trigger JSON: {json_data}")
                            print(type(json_data))  # should be a dictionary
                            self.maze_data_signal.emit(json_data)
                        except json.JSONDecodeError:
                            print("[MazeDataReceiverThread] Received malformed JSON.")
                    except socket.timeout:
                        continue  # Check if still running
                    except ConnectionResetError:
                        print("[MazeDataReceiverThread] Connection reset by peer.")
                        break
                    except OSError as e:
                        if not self.is_running:
                            break  # Socket was closed, exit the loop
                        else:
                            print(f"[MazeDataReceiverThread] OSError during recv: {e}")
                            break

                # Close client socket after disconnection
                if self.client_socket:
                    try:
                        self.client_socket.close()
                    except Exception as e:
                        print(f"[MazeDataReceiverThread] Exception while closing client socket: {e}")
                    self.client_socket = None

        except Exception as e:
            print(f"[MazeDataReceiverThread] Exception: {e}")
            QMessageBox.critical(None, "Maze Connection Error", f"An error occurred: {e}")
            self.is_running = False

    def stop(self):
        print("[MazeDataReceiverThread] Stopping thread.")
        self.is_running = False
        # Close client socket first to unblock recv
        if self.client_socket:
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
                self.client_socket.close()
            except Exception as e:
                print(f"[MazeDataReceiverThread] Exception during client socket shutdown: {e}")
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception as e:
                print(f"[MazeDataReceiverThread] Exception during server socket close: {e}")
        self.quit()
        self.wait()
        print("[MazeDataReceiverThread] Thread stopped.")

# Define the main ClientWindow
class ClientWindow(QMainWindow):
    def __init__(self, board_id=BoardIds.SYNTHETIC_BOARD, params=None, maze_host='localhost', maze_port=65432):
        super().__init__()
        print("[ClientWindow] Initializing ClientWindow.")
        self.setWindowTitle("NeuroNavScore Client Application")
        self.setGeometry(100, 100, 1200, 800)  # Adjusted size for better visualization

        # Initialize Test Control Variables
        self.test_running = False
        self.test_paused = False
        self.test_start_time = None
        self.test_duration = 10  # default duration in seconds
        self.elapsed_time = 0
        self.pass_fail_result = None
        self.score = 0

        # Initialize Data Structures
        self.eeg_channels = 4
        self.board_id = board_id
        graph_window_seconds = 5
        buffer_size = graph_window_seconds * BoardShim.get_sampling_rate(self.board_id) + 200
        self.eeg_data = np.zeros((self.eeg_channels, buffer_size))
        self.t = np.zeros(buffer_size)
        self.ticks = {}

        # Initialize UI
        self.init_ui()

        # Initialize Data Acquisition Thread
        self.init_data_acquisition(board_id, params)

        # Initialize Maze Data Receiver Thread
        self.init_maze_data_receiver(maze_host, maze_port)

        # Initialize Timer for Test Monitoring
        self.test_timer = QTimer()
        self.test_timer.timeout.connect(self.monitor_test)

    def init_ui(self):
        print("[ClientWindow] Setting up UI.")
        # Main Layout
        main_layout = QVBoxLayout()

        # Title Label
        title_label = QLabel("NeuroNavScore")
        title_label.setStyleSheet("font-size: 36px; font-weight: bold;")
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
        self.eeg_graph = pg.PlotWidget()
        self.eeg_graph.setMouseEnabled(x=False, y=False)
        # self.eeg_graph.setYRange(-150, 150)
        self.eeg_graph.showGrid(x=True, y=True)
        self.eeg_graph.addLegend()
        self.eeg_graph.hideAxis("bottom")

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
        self.start_test_button.setToolTip("Click to start the navigation test.")

        self.pause_test_button = QPushButton("Pause Test")
        self.pause_test_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.pause_test_button.clicked.connect(self.pause_test)
        self.pause_test_button.setEnabled(False)
        self.pause_test_button.setToolTip("Click to pause or resume the test.")

        self.stop_test_button = QPushButton("Stop Test")
        self.stop_test_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.stop_test_button.clicked.connect(self.stop_test)
        self.stop_test_button.setEnabled(False)
        self.stop_test_button.setToolTip("Click to stop the test prematurely.")

        self.reset_test_button = QPushButton("Reset Test")
        self.reset_test_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.reset_test_button.clicked.connect(self.reset_test)
        self.reset_test_button.setEnabled(False)
        self.reset_test_button.setToolTip("Click to reset all test data.")

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
        self.progress_bar.setFormat(f"Test Progress: 0/{self.test_duration}/s")
        self.progress_bar.setStyleSheet("QProgressBar::chunk {background-color: #05B8CC;}")

        # Performance Indicators Group
        performance_group = QGroupBox("Performance Indicators")
        performance_layout = QHBoxLayout()

        self.score_label = QLabel("Visuospatial Processing Score: 0 - N/A")
        self.score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: gray;")
        self.score_label.setAlignment(Qt.AlignCenter)

        performance_layout.addWidget(self.score_label)
        performance_group.setLayout(performance_layout)

        # Export Results Button
        self.export_button = QPushButton("Export Results")
        self.export_button.setStyleSheet("font-size: 14px; padding: 8px;")
        self.export_button.clicked.connect(self.export_results)
        self.export_button.setEnabled(False)
        self.export_button.setToolTip("Click to export test results as a CSV file.")

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

    def init_data_acquisition(self, board_id, params):
        print("[ClientWindow] Initializing data acquisition thread.")
        self.data_thread = DataAcquisitionThread(board_id=board_id, params=params)
        self.data_thread.eeg_data_signal.connect(self.update_eeg_data)
        self.data_thread.start()
        self.status_label.setText("Connection Status: EEG Connected")
        self.status_label.setStyleSheet("font-weight: bold; color: green;")

    def init_maze_data_receiver(self, host, port):
        print("[ClientWindow] Initializing maze data receiver thread.")
        self.maze_thread = MazeDataReceiverThread(host=host, port=port)
        self.maze_thread.maze_data_signal.connect(self.process_maze_data)
        self.maze_thread.start()

    @Slot(list)
    def update_eeg_data(self, data):
        eeg_data, t = data
        # Update the t
        self.t = np.roll(self.t, -len(t), 0)
        self.t[-len(t):] = t
        # Update the EEG data queues
        for i in range(self.eeg_channels):
            self.eeg_data[i] = np.roll(self.eeg_data[i], -len(eeg_data[i]), 0)
            self.eeg_data[i, -len(eeg_data[i]):] = eeg_data[i]

        sr = BoardShim.get_sampling_rate(self.board_id)
        # Update the EEG graphs
        for i, curve in enumerate(self.curves):
            data = self.eeg_data[i].copy()
            # DataFilter.remove_environmental_noise(data, sr, NoiseTypes.SIXTY.value)
            # DataFilter.detrend(data, DetrendOperations.CONSTANT.value)
            # DataFilter.perform_bandpass(data, sr, 4, 8, 4, FilterTypes.BUTTERWORTH, 0)
            curve.setData(x=self.t[200:], y=data[200:])
        
        # Update the ticks
        for tick in self.ticks.keys():
            if tick < self.t[200] and self.ticks[tick] is not None:
                self.eeg_graph.removeItem(self.ticks[tick])  # delete the inf line
            if tick <= self.t[-1] and self.ticks[tick] is None:
                self.ticks[tick] = self.eeg_graph.addLine(x=tick, pen=pg.mkPen('r', width=5))
                self.ticks[tick].setZValue(20)
        # Update the visuospatial processing score
        # Replace this with your actual scoring logic
        # eeg_sum = sum(abs(val) for val in eeg_data)
        # self.score = min(int(eeg_sum), 100)  # Clamp score to 100
        # self.score_label.setText(f"Visuospatial Processing Score: {self.score} - N/A")
        # print(f"[ClientWindow] Updated Score: {self.score}")

    @Slot(dict)
    def process_maze_data(self, maze_data):
        # Handle maze data received from the maze application
        # Example data structure:
        # {
        #     'triggerID': 'T3',
        # }
        print(f"[ClientWindow] Processing maze data: {maze_data}")
        event = maze_data.get('triggerID', '')
        if event:
            self.insert_marker()
            tick = time.time()
            self.ticks[tick] = None

    def insert_marker(self, id=1):
        if hasattr(self, 'data_thread'):
            self.data_thread.board.insert_marker(id)

    def update_test_duration(self, value):
        print(f"[ClientWindow] Test duration updated to {value} seconds.")
        self.test_duration = value
        if not self.test_running:
            self.progress_bar.setMaximum(self.test_duration)
            self.progress_bar.setFormat(f"Test Progress: 0/{self.test_duration}/s")

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
        self.progress_bar.setFormat(f"Test Progress: 0/{self.test_duration}/s")
        print(f"[ClientWindow] Test duration set to {self.test_duration} seconds.")

        # Initialize Test Variables
        self.test_running = True
        self.test_paused = False
        self.test_start_time = time.time()
        self.elapsed_time = 0
        self.pass_fail_result = None
        self.score = 0
        self.score_label.setText("Visuospatial Processing Score: 0 - N/A")
        self.score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: gray;")
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
        self.score_label.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {color};")
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
        self.score_label.setText("Visuospatial Processing Score: 0 - N/A")
        self.score_label.setStyleSheet("font-size: 24px; font-weight: bold; color: gray;")
        self.progress_bar.setValue(0)
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
                        "Visuospatial Processing Score",
                        "Timestamp"
                    ])
                    writer.writerow([
                        self.name_input.text().strip(),
                        self.age_input.text().strip(),
                        self.pass_fail_result,
                        self.score,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ])
                QMessageBox.information(self, "Export Successful", f"Results exported to {file_path}")
                print(f"[ClientWindow] Results exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"An error occurred while exporting results:\n{e}")
                print(f"[ClientWindow] Export failed: {e}")

    def monitor_test(self):
        self.elapsed_time += 1
        self.progress_bar.setValue(self.elapsed_time)
        self.progress_bar.setFormat(f"Test Progress: {self.elapsed_time}/{self.test_duration}/s")
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
        self.score_label.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {color};")
        self.score_label.setText(f"Visuospatial Processing Score: {self.score} - {self.pass_fail_result}")

        # Enable Export Button
        if self.pass_fail_result in ['Pass', 'Fail']:
            self.export_button.setEnabled(True)

        # Inform Clinician
        QMessageBox.information(self, "Test Completed", f"Navigation test completed with result: {self.pass_fail_result}")
        print(f"[ClientWindow] Test completed with result: {self.pass_fail_result}")

    def closeEvent(self, event):
        print("[ClientWindow] Closing application.")
        if hasattr(self, 'data_thread'):
            self.data_thread.stop()
        if hasattr(self, 'maze_thread'):
            self.maze_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Define BrainFlow input parameters
    params = BrainFlowInputParams()
    params.serial_port = '/dev/cu.usbmodem11'  # Update this to your actual serial port

    # Choose the appropriate board ID
    # For example, BoardIds.CYTON_BOARD for Cyton boards
    # Refer to BrainFlow documentation for supported board IDs
    board_id = BoardIds.GANGLION_BOARD # Replace with your actual board ID, e.g., BoardIds.CYTON_BOARD
    
    # Define Maze Server Parameters
    maze_host = '0.0.0.0'  # The UI will listen on this host
    maze_port = 12345         # The UI will listen on this port

    client = ClientWindow(board_id=board_id, params=params, maze_host=maze_host, maze_port=maze_port)
    client.show()
    sys.exit(app.exec())
