"""Script to detect objects that are being shaken in front of the camera."""
import cv2
import math
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

        frames = stabilize(frames)

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

def stabilize(frames):
    # Accumulated frame transforms.
    acc_dx = 0
    acc_dy = 0
    acc_da = 0

    acc_transform = numpy.zeros((3, 3), numpy.float32)
    acc_transform[0, 0] = 1
    acc_transform[1, 1] = 1
    acc_transform[2, 2] = 1

    # Highest translations (left/right, top/bottom), used to compute a mask
    min_acc_dx = 0
    max_acc_dx = 0
    min_acc_dy = 0
    max_acc_dy = 0

    stabilized = []

    prev = frames[0]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)

    debug_before_writer = cv2.VideoWriter("/tmp/debug_raw.avi", cv2.VideoWriter_fourcc(*"DIVX"), 16, (WIDTH, HEIGHT));
    debug_after_writer = cv2.VideoWriter("/tmp/debug_stabilized.avi", cv2.VideoWriter_fourcc(*"DIVX"), 16, (WIDTH, HEIGHT));

    # Stabilize image, most likely introducing borders.
    stabilized.append(prev)
    for cur in frames[1:]:
        debug_before_writer.write(cur)
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)
        cv2.imshow('gray', prev_gray)

        prev_corner = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=.01, minDistance=10) # FIXME: What are these constants?
        if not (prev_corner is None):
            # FIXME: Really, what should we do if `prev_corner` is `None`?
            cur_corner, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, prev_corner, None)
            should_copy = [bool(x) for x in status]

            # weed out bad matches
            # FIXME: I'm sure that there is a more idiomatic way to do this in Python
            corners = len(should_copy)
            prev_corner2 = numpy.zeros((corners, 1, 2), numpy.float32)
            cur_corner2 = numpy.zeros((corners, 1, 2), numpy.float32)

            j = 0
            for i in range(len(status)):
                if status[i]:
                    prev_corner2[j] = prev_corner[i]
                    cur_corner2[j] = cur_corner[i]
                    j += 1
            prev_corner = None
            cur_corner = None


            # Compute transformation between frames, as a combination of translations, rotations, uniform scaling.
            transform = cv2.estimateRigidTransform(prev_corner2, cur_corner2, False)
            if transform is None:
                print("stabilize: could not find transform, skipping frame")
            else:
                dx = transform[0, 2]
                dy = transform[1, 2]
                da = math.atan2(transform[1, 0], transform[0, 0])

                result = None

                if dx == 0. and dy == 0.:
                    print("stabilize: dx and dy are 0")
                    result = cur
                else:
                    acc_dx += dx
                    if acc_dx > max_acc_dx:
                        max_acc_dx = acc_dx
                    elif acc_dx < min_acc_dx:
                        min_acc_dx = acc_dx

                    acc_dy += dy
                    if acc_dy > max_acc_dy:
                        max_acc_dy = acc_dy
                    elif acc_dy < min_acc_dy:
                        min_acc_dy = acc_dy

                    acc_da += da

                    padded_transform = numpy.zeros((3, 3), numpy.float32)
                    for i in range(2):
                        for j in range(3):
                            padded_transform[i,j] = transform[i,j]
                    padded_transform[2, 2] = 1
                    acc_transform = numpy.dot(acc_transform, padded_transform)

                    print("stabilize: current transform\n %s" % transform)
                    print("stabilize: padded transform\n %s" % padded_transform)
                    print("stabilize: full transform\n %s" % acc_transform)
                    print("stabilize: resized full transform\n %s" % acc_transform[0:2, :])
                    result = cv2.warpAffine(cur, acc_transform[0:2,:], (WIDTH, HEIGHT), cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP )
                stabilized.append(result)

                debug_after_writer.write(result)
        else:
            print("stabilize: could not find prev_corner, skipping frame")

        prev = cur
        prev_gray = cur_gray

    # Now crop all images to remove these borders.
    cropped = []
    for frame in stabilized:
#        cropped.append(frame[max_acc_dx:max_acc_dy, min_acc_dx:min_acc_dy])
        cropped.append(frame)
        # FIXME: Actually crop

    return cropped

if __name__ == '__main__':
    sys.exit(main())