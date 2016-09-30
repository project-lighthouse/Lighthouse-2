"""Script to detect objects that are being shaken in front of the camera."""
import cv2
import numpy
import sys

WIDTH = 320
HEIGHT = 200

# Number of frames during which to run detection. Divide by ~16 to obtain
# a number of seconds.
FRAMES_TO_DETECT = 30

def main():
    """Main entry point for the script."""
    source = 0
    if len(sys.argv) >= 2:
        source = sys.argv[1]
    print("Lightouse 2 starting with source %s"  % source)

    cap = cv2.VideoCapture(source)
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source')
        return -1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    backgroundSubstractor = cv2.createBackgroundSubtractorMOG2()

    # Record pixels where we have had most motion.
    collector = None
    prev = None
    frame = None
    detecting = False
    detection_countdown = 0
    force_start = True

    while(True):
        # Capture frame-by-frame
        ret, current = cap.read()

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord(' ') or force_start:
            force_start = False
            detecting = True
            collector = numpy.zeros((HEIGHT, WIDTH), numpy.uint64)
            detection_countdown = FRAMES_TO_DETECT

        if not ret:
            # Somehow, we failed to capture the frame.
            continue
        prev = frame
        frame = current

        # Our operations on the frame come here
#        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if detecting:
            motion = backgroundSubstractor.apply(frame) # FIXME: Is this the right subtraction?
            positive = cv2.bitwise_and(motion, 255)
            negative = cv2.bitwise_and(motion, 127)
#            filtered = cv2.bitwise_and(gray, positive)
            collector = collector + positive
            detection_countdown -= 1
            if detection_countdown <= 0:
                # Detection is over, time to extract/show the result.
                detecting = False

                # FIXME: We are only interested in pixels for which we have
                # seen motion recently AND show up as active in collector.
                threshold = numpy.median(collector)
                # ret, collector = cv2.threshold(collector, threshold, 255, cv2.THRESH_BINARY)

#                display = cv2.bitwise_and(collector, frame)

                # Display where we have seen most motion during the countdown.
#                cv2.imshow('result', display)
                cv2.imshow('collector', collector)
                cv2.imshow('motion', motion)


        # Display the resulting frame
        cv2.imshow('frame', frame)


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    pass

if __name__ == '__main__':
    sys.exit(main())