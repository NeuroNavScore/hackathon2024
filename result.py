from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
import pyqtgraph as pg
from brainflow.data_filter import DataFilter
from brainflow.board_shim import BoardShim, BoardIds

import mne
import numpy as np


class Results(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Create a plot widget
        self.plotWidget = pg.PlotWidget(title="Theta Activity")
        layout.addWidget(self.plotWidget)

        # Sample data for the bar graph
        x = [0, 1]
        y = self.calculate_theta()

        # Create a bar graph item
        barGraph = pg.BarGraphItem(x=x, height=y, width=0.6, brush='b')
        self.plotWidget.addItem(barGraph)
        ax = self.plotWidget.getAxis('bottom')
        ax.setTicks([[(0, 'Easy'), (1, 'Hard')]])

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def calculate_theta(self):
        '''
        returns theta: list of theta values for each trial
        '''
        # get the data from eeg_data.csv
        data = DataFilter.read_file('eeg_data.csv')
        # calculate theta values
        board_id = BoardIds.GANGLION_BOARD
        channels = BoardShim.get_eeg_channels(board_id)
        sr = BoardShim.get_sampling_rate(board_id)

        channel_data = data[channels, :]
        channel_data /= 1e6  # uV to V (ganglion reports in uV)
        ch_names = [f"EEG{i}" for i in range(channel_data.shape[0])]
        ch_types = ["eeg"] * channel_data.shape[0]
        info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sr)
        raw = mne.io.RawArray(channel_data, info)  # input is (n_channels, n_samples)
        tmin, tmax = -0.5, 1  # in seconds
        markers = data[BoardShim.get_marker_channel(board_id), :]  # (n_samples, )
        theta = [0, 0]

        # Easy event markers
        power = theta_power(raw, markers, 1, tmin, tmax)
        theta[0] = power
        power = theta_power(raw, markers, 2, tmin, tmax)
        theta[1] = power
        return theta

def theta_power(raw, markers, event_id, tmin, tmax):
    easy_event_indices = np.argwhere(markers == event_id)
    zeroes = np.zeros(easy_event_indices.shape)
    event_labels = markers[easy_event_indices]
    events = np.concatenate([easy_event_indices, zeroes, event_labels], axis=1).astype(int)
    epochs = mne.Epochs(raw, events, {f"{event_id}": event_id}, tmin, tmax)
    epochs.plot(n_epochs=2, n_channels=4, events=True, scalings="auto")

    spectrum = epochs.compute_psd()
    psd, freqs = spectrum.get_data(return_freqs=True)
    # find nearest indices in freq that match up to 4 and 8 Hz
    i4 = find_nearest(freqs, 4)
    i8 = find_nearest(freqs, 8)

    # calculate theta power
    power = np.mean(psd[:, :, i4:i8])
    return power

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

if __name__ == "__main__":
    app = QApplication([])

    w = Results()
    w.show()

    app.exec()