import cv2
import time

# FIXME: Temporary method that just captures frames without any processing.
def capture(args):
    cap = cv2.VideoCapture(args['video_source'])

    if cap is None or not cap.isOpened():
        print('Error: unable to open video source')
        return -1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args['video_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args['video_height'])
    cap.set(cv2.CAP_PROP_FPS, args['video_fps'])

    frames = []
    for frame_index in range(0, args['acquisition_keep_objects']):
        frame_read_time = time.time()
        ret, frame = cap.read()
        frames.append(frame)
        if args['verbose']:
            print('Frame %s has been acquired in %s seconds.' % (frame_index, time.time() - frame_read_time))

    return frames

