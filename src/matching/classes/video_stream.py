from threading import Thread
import cv2
import os
import signal
import subprocess
import time


class VideoStream:
    def __init__(self, src, width, height, fps):
        self._src = src
        self._width = width
        self._height = height
        self._fps = fps

        self._stream = None

        self._started = False

    def is_started(self):
        return self._started and self._stream.isOpened()

    def start(self):
        self._stream = cv2.VideoCapture(self._src)
        self._stream.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._stream.set(cv2.CAP_PROP_FPS, self._fps)

        self._started = True

        #thread = Thread(target=self.update, args=())
        #thread.daemon = True
        #thread.start()

        return self

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while True:
            # If the thread indicator variable is set, stop the thread.
            if not self._started:
                return

            self._stream.grab()

    def read(self):
        return self._stream.read()

    def stop(self):
        self._started = False

        self._stream.release()
        self._stream = None
