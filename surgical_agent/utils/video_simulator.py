import cv2
import threading
import time
class LiveCameraSimulator:
    """Simulates a live camera feed from a video file with threaded reading."""

    def __init__(self, path):
        """
        Initialize the video stream simulator.

        Args:
            path (str): Path to the video file.
        """
        self.stream = cv2.VideoCapture(path)
        self.fps = self.stream.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0:
            self.fps = 30
        self.frame_time = 1.0 / self.fps
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        """
        Start the background thread for reading frames.

        Returns:
            LiveCameraSimulator: Self for method chaining.
        """
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stopped = True
            else:
                (self.grabbed, self.frame) = self.stream.read()
            time.sleep(0.001)

    def read(self):
        """
        Get the latest frame from the stream.

        Returns:
            numpy.ndarray: Current frame, or None if stream is closed.
        """
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()
