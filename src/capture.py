import cv2
import time

# FIXME: Temporary method that just captures frames without any processing.
# XXX
# I'd like to open the camera earlier and free it after a period of idle time
# so that I can grab frames more quickly.
# But in order to do that, I've got to have a background thread calling
# grab() all the time on the camera, because otherwise we get
# out of date frames from the video buffer.
def capture(options):
    camera = cv2.VideoCapture(options.video_source)

    if camera is None or not camera.isOpened():
        print("Error: can't open camera")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, options.video_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, options.video_height)
    camera.set(cv2.CAP_PROP_FPS, options.video_fps)

    frame_read_time = time.time()
    ret, frame = camera.read()
    camera.release()

    if options.verbose:
        print('Frame acquired in %s seconds.' % (time.time() - frame_read_time))

    return frame

