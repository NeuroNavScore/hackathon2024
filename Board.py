import sys
import time
import numpy as np
import os
from datetime import datetime  # Import datetime for timestamps
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow import DataFilter

# Define the Board class to handle BrainFlow session
class Board:
    def __init__(self, board_id, params = BrainFlowInputParams()):
        self.board_id = board_id
        self.params = params
        self.board = None

    def __del__(self):
        if self.board and self.board.is_prepared():
            self.board.release_session()
            print("Board session released in destructor.")

    def start_session(self):
        while True:
            try:
                # Initialize and start the board session
                self.board = BoardShim(self.board_id, self.params)
                self.board.prepare_session()
                self.board.start_stream()
                print("Session started.")
                return
            
            except Exception as e:
                print(f"Error during board session: {e}")
            print("Retrying...")
            
    def stop_session(self):
        try:
            if self.board is not None:
                # Stop the board stream and release the session
                self.board.stop_stream()
                self.board.release_session()

                print("Recording stopped.")
        except Exception as e:
            print(f"Error during stopping session: {e}")

    def save_data(self):
        try:
            # Get the board data (for example, from all available channels)
            data = self.board.get_board_data()
            DataFilter.write_file(data, self.fname, 'a')
            # print("Appended new EEG data to file.")

        except Exception as e:
            print(f"Error during data recording: {e}")

    def insert_marker(self, id):
        self.board.insert_marker(id)

def main():
    params = BrainFlowInputParams()
    params.serial_port = '/dev/cu.usbmodem11'
    board = Board(
        BoardIds.SYNTHETIC_BOARD, 
        params=params
    )
    
    try:
        # Start session
        board.start_session()
        # Collect data every 2 seconds
        while True:
            time.sleep(2)
            print(board.board.get_board_data())
    except KeyboardInterrupt:
        # Stop session on interrupt
        print("\nStopping session...")
        board.stop_session()

if __name__ == "__main__":
    main()
