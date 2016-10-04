"""Script to detect objects that are being shaken in front of the camera."""
import cv2
import numpy
import sys

WIDTH = 320
HEIGHT = 200
BLUR = (5, 5)

# Number of frames during which to run detection. Divide by ~16 to obtain
# a number of seconds.
FRAMES_BUFFER_SIZE = 60

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

    idle = True
    force_start = True

    # A buffer holding the frames. It will hold up to FRAMES_BUFFER_SIZE framesselfself.
    frames = None

    while(True):
        # Capture frame-by-frame.
        ret, current = cap.read()

        key = cv2.waitKey(1) & 0xFF
        # <q> or <Esc>: quit
        if key == 27 or key == ord('q'):
            break
        # <spacebar> or `force_start`: start detecting.
        elif key == ord(' ') or force_start:
            force_start = False
            idle = False
            frames = []

        if not ret:
            # Somehow, we failed to capture the frame.
            continue

        # Display the current frame
        cv2.imshow('frame', current)
        cv2.moveWindow('frame', 0, 0)

        if idle:
            continue

        if len(frames) < FRAMES_BUFFER_SIZE:
            # We are not done buffering.
            frames.append(current)
            continue

        # At this stage, we are done buffering. Stop recording, start processing.
        idle = True

        # FIXME: Stabilize image

        # Extract foreground
        foreground = None
        for frame in frames:
            foreground = backgroundSubstractor.apply(frame) # FIXME: Is this the right subtraction?

        # Smoothen a bit the mask to get back some of the missing pixels
        mask = cv2.blur(cv2.bitwise_and(foreground, 255), BLUR)
        ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        cv2.imshow('tracking', mask)
        cv2.moveWindow('tracking', WIDTH + 32, HEIGHT + 32)

        tracking = cv2.bitwise_and(color_mask, current)
        cv2.imshow('objects', tracking)
        cv2.moveWindow('objects', 0, HEIGHT + 32)
        cv2.imwrite('/tmp/object.png', tracking)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    pass

if __name__ == '__main__':
    sys.exit(main())