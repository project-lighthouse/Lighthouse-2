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
        print('Error: unable to open video source')
        return -1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    backgroundSubstractor = cv2.createBackgroundSubtractorMOG2()

    # Record pixels where we have had most motion.
    detecting = False
    detection_countdown = 0
    force_start = True

    while(True):
        # Capture frame-by-frame
        ret, current = cap.read()

        key = cv2.waitKey(1) & 0xFF
        # <q> or <Esc>: quit
        if key == 27 or key == ord('q'):
            break
        # <spacebar> or `force_start`: start detecting.
        elif key == ord(' ') or force_start:
            force_start = False
            detecting = True
            detection_countdown = FRAMES_TO_DETECT

        if not ret:
            # Somehow, we failed to capture the frame.
            continue

        # Display the current frame
        cv2.imshow('frame', current)
        cv2.moveWindow('frame', 0, 0)

        # Our operations on the frame come here
        if detecting:
            foreground = backgroundSubstractor.apply(current) # FIXME: Is this the right subtraction?
            mask = cv2.bitwise_and(foreground, 255)
            color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            detection_countdown -= 1

            cv2.imshow('tracking', mask)
            cv2.moveWindow('tracking', WIDTH + 32, HEIGHT + 32)

            tracking = cv2.bitwise_and(color_mask, current)
            cv2.imshow('objects', tracking)
            cv2.moveWindow('objects', 0, HEIGHT + 32)

            if detection_countdown <= 0:
                # Detection is over, time to extract/show the result.
                detecting = False
                extrapolated = cv2.inpaint(tracking, cv2.bitwise_not(mask), 3, cv2.INPAINT_TELEA)

                cv2.imshow('extrapolated', extrapolated)
                cv2.moveWindow('extrapolated', WIDTH + 32, 0)
                cv2.imwrite('/tmp/object.png', tracking)
                cv2.imwrite('/tmp/inpainted.png', extrapolated)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    pass

if __name__ == '__main__':
    sys.exit(main())