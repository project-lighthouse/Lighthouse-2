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
        self._raspivid_process = None
        self._netcat_process = None

        self._started = False
        self._paused = False

    def is_started(self):
        return self._started and self._stream.isOpened()

    def start(self):
        if self._src == 'raspivid':
            self._raspivid_process = subprocess.Popen(['raspivid', '-t', '0', '-w', str(self._width), '-h',
                                                       str(self._height), '-fps', str(self._fps), '-s', '-o', '-'],
                                                      stdout=subprocess.PIPE)
            self._netcat_process = subprocess.Popen(['nc', '-l', '-p', '2222'], stdin=self._raspivid_process.stdout)
            src = 'tcp://0.0.0.0:2222'
        else:
            src = self._src

        self._stream = cv2.VideoCapture(src)
        self._stream.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._stream.set(cv2.CAP_PROP_FPS, self._fps)

        self._started = True

        thread = Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()

        return self

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while True:
            # If the thread indicator variable is set, stop the thread.
            if not self._started:
                return

            self._stream.grab()

    def read(self):
        return self._stream.retrieve()

    def pause(self):
        # Pause only works for the 'raspivid' setup.
        if self._raspivid_process is None or self._paused:
            return

        os.kill(self._raspivid_process.pid, signal.SIGUSR1)

    def resume(self):
        # Resume only works for the 'raspivid' setup.
        if self._raspivid_process is None or not self._paused:
            return

        os.kill(self._raspivid_process.pid, signal.SIGUSR1)

    def write_frame(self, path):
        paused = self._paused

        if paused:
            self.resume()

        ret, frame = self.read()
        cv2.imwrite(path, frame)

        if paused:
            self.pause()

    def stop(self):
        self._started = False
        self._paused = False

        self._stream.release()
        self._stream = None

        if self._raspivid_process is not None:
            self._netcat_process.terminate()
            self._netcat_process = None

            self._raspivid_process.terminate()
            self._raspivid_process = None
