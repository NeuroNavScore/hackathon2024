from Board import Board
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BrainFlowError
import numpy as np

class Processor(Board):
    def __init__(self, board_id, params, channels=None):
        super().__init__(board_id, params)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        if channels is None:
            self.channels = BoardShim.get_exg_channels(self.board_id)
        else:
            self.channels = channels  # Use only 1 channel
        # self.channels = [self.channels[0]]  # Use only 1 channel
        n_channels = len(self.channels)

    def process(self):
        new_raw = self.raw()
        new_filtered = self.filter(new_raw)
        return new_raw, new_filtered

    def raw(self):
        '''
        -> raw
        '''
        new_data = self.board.get_board_data()[self.channels, -self.buffer_size:]  # Get as much data as possible until buffer is full
        if new_data.shape[1] == 0:
            return new_data  # Return empty array if no new data
        n = new_data.shape[1]  # Number of new samples
        for channel in range(len(self.channels)):
            # Shift and fill the buffer
            self.raw_signal[channel] = np.roll(self.raw_signal[channel], -n, axis=0)  # Shift the buffer
            self.raw_signal[channel, -n:] = new_data[channel]  # Fill the buffer with new data
        return new_data

    def filter(self, new_raw):
        '''
        raw -> filtered
        '''
        n = new_raw.shape[1]  # Number of new samples
        # new_filtered = new_raw.copy()  # Copy the data to avoid modifying the original
        new_filtered = self.raw_signal.copy()  # include the entire buffer to avoid boundary effects
        for channel in range(len(self.channels)):
            # DataFilter.detrend(data, DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandstop(new_filtered[channel], self.sampling_rate, 0, 60, 4,
                                        FilterTypes.BUTTERWORTH, 0)
            # DataFilter.perform_lowpass(data, self.sampling_rate, 360, 4,
            #                             FilterTypes.BUTTERWORTH, 0)
            DataFilter.perform_bandpass(new_filtered[channel], self.sampling_rate, 90, 330, 4,
                                        FilterTypes.BUTTERWORTH, 0)
            # DataFilter.perform_bandpass(new_filtered[channel], self.sampling_rate, 20, 500, 4,
            #                             FilterTypes.BUTTERWORTH, 0)
            # DataFilter.perform_bandpass(new_filtered[channel], self.sampling_rate, 20, 500, 2,
            #                 FilterTypes.BUTTERWORTH, 0)
            # Shift and fill the buffer
            self.filtered_signal[channel] = np.roll(self.filtered_signal[channel], -n, axis=0)
            self.filtered_signal[channel, -n:] = new_filtered[channel, -n:]

        new_filtered = new_filtered[:, -n:]  # Return only the new data
        return new_filtered