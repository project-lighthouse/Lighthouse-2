"""Script to detect objects that are being shaken in front of the camera."""
import cv2
import sys

def main():
    """Main entry point for the script."""
    if len(sys.argv) < 2:
        print("Expected a video name")
        return -1
    print("Lightouse 2 starting with video %s"  % (sys.argv[1]))

    cap = cv2.VideoCapture(0)
    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source')
        return -1
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 200)

    backgroundSubstractor = cv2.createBackgroundSubtractorMOG2()

    while(True):
        # Placeholder - for the time being, we're converting to gray.

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = backgroundSubstractor.apply(gray)

        # Display the resulting frame
        cv2.imshow('frame',gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    pass

if __name__ == '__main__':
    sys.exit(main())