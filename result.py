from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
import pyqtgraph as pg
from brainflow.data_filter import DataFilter
from brainflow.board_shim import BoardShim, BoardIds

import mne


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
        ax.setTicks([[(0, 'N'), (1, 'S')]])

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
        channels = BoardShim.get_eeg_channels(BoardIds.GANGLION_BOARD)
        sr = BoardShim.get_sampling_rate(BoardIds.GANGLION_BOARD)
        result = DataFilter.get_avg_band_powers(data, channels, sr, True)
        theta = result[0][1]

        return theta
if __name__ == "__main__":
    app = QApplication([])

    w = Results()
    w.show()

    app.exec()