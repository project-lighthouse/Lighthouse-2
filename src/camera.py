import logging
from Queue import Queue
from threading import Thread, Event, Timer
import cv2


#
# This class wraps cv2.VideoCapture, and the capture() method returns
# still frames from a video source. To improve speed, the VideoCapture
# object can be is created in advance by calling start(). You can call
# shutdown() to release the camera and its thread, but you don't need
# to do this: it will be automatically shut down after a specified
# period of inactivity.
#
# In order to make this work, however, this class runs a thread that
# repeatedly calls grab() on the VideoCapture object to so that old
# frames are not buffered up. Because of this background thread, calling
# capture() returns the latest frame.
#
class Camera(object):
    def __init__(self, source, width=640, height=480, fps=15, shutdown_time=15):
        self.logger = logging.getLogger(__name__)

        # Parameters for the VideoCapture object
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps

        # How long we wait before shutting the VideoCapture object and its
        # thread down
        self.shutdown_time = shutdown_time

        # The timer we use to stop the camera thread
        self.shutdown_timer = None

        # This is the thread that controls the camera
        self.thread = None

        # We request a picture from the camera thread by setting this flag
        self.picture_flag = Event()

        # The camera thread returns the picture by putting it on this queue
        self.queue = Queue()

        # We set this flag when we want the camera thread to release the
        # camera and exit
        self.shutdown_flag = Event()

    # Return the most recently grabbed frame from the camera, starting
    # the camera if necessary.
    def capture(self):
        self.logger.debug("Capture is requested.")
        self.start()             # start the camera if necessary
        self.picture_flag.set()  # ask for a picture
        image = self.queue.get() # block until it appears on the queue
        return image

    # Call this to start the camera. If you know you'll need it soon,
    # you can call this in advance of capture to speed things up a bit.
    def start(self):
        self._reset_shutdown_timer()
        if self.thread:
            self.logger.debug("Capture thread is already started.")
            return
        self.thread = Thread(name="camera-thread", target=self._thread)
        self.thread.daemon = True
        self.thread.start()
        self.logger.debug("Capture thread is started.")

    # Call this to shut the camera (and its thread) down.
    # It will automatically shutdown after self.shutdown_time seconds
    # so you shouldn't have to call it manually.
    def shutdown(self):
        self.shutdown_flag.set()
        self.shutdown_timer.cancel()
        self.shutdown_timer = None

    def _reset_shutdown_timer(self):
        if self.shutdown_timer:
            self.shutdown_timer.cancel()
        self.shutdown_timer = Timer(self.shutdown_time, self.shutdown)
        self.shutdown_timer.start()

    def _thread(self):
        camera = cv2.VideoCapture(self.source)

        # FIXME: We should do something about it, either crash entire program
        # or use a loop with few seconds delay that will be constantly trying
        # to open camera, that will crash program after several attempts anyway.
        if camera is None or not camera.isOpened():
            logging.error("Can't open camera.")
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        camera.set(cv2.CAP_PROP_FPS, self.fps)

        self._reset_shutdown_timer()

        while not self.shutdown_flag.is_set():
            camera.grab()
            if self.picture_flag.is_set():
                self.logger.debug("Retrieving a new frame.")
                _, image = camera.retrieve()
                self._reset_shutdown_timer()
                self.queue.put(image)
                self.picture_flag.clear()

        # Exiting
        self.thread = None
        self.shutdown_flag.clear()
        camera.release()

        self.logger.debug("Camera thread is stopped.")
