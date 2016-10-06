"""Script to detect objects that are being shaken in front of the camera."""
import argparse
import math
import sys

import cv2
import numpy

parser = argparse.ArgumentParser(description='Detect/compare objects being shaken in front of the camera.')
parser.add_argument('--source', help='Video to use (default: built-in cam)', default=0)
parser.add_argument('--dump-raw', help='Write raw captured video to this file (default: none)', default=None)
parser.add_argument('--dump-stabilized', help='Write stabilized video to this file (default: none)', default=None)
parser.add_argument('--dump-object', help='Write captured object to this file (default: none)', default=None)
parser.add_argument('--dump-mask', help='Write mask to this file (default: none)', default=None)
parser.add_argument('--width', help='Video width (default: 320)', default=320, type=int)
parser.add_argument('--height', help='Video height (default: 200)', default=200, type=int)
parser.add_argument('--blur', help='Blur radius (default: 15)', default=15, type=int)
parser.add_argument('--buffer', help='Number of frames to capture before proceeding (default: 60)', default=60, type=int)

parser.add_argument('--autostart', help='Start processing immediately (default).', dest='autostart', action='store_true')
parser.add_argument('--no-autostart', help='Do not start processing immediately.', dest='autostart', action='store_false')
parser.set_defaults(autostart=True)

parser.add_argument('--autoexit', help='Quit once the process is complete (default).', dest='autoexit', action='store_true')
parser.add_argument('--no-autoexit', help='Do not quit once the process is complete.', dest='autoexit', action='store_false')
parser.set_defaults(autoexit=True)

parser.add_argument('--show', help='Display videos (default).', dest='show', action='store_true')
parser.add_argument('--no-show', help='Do not display videos.', dest='show', action='store_false')
parser.set_defaults(show=True)

parser.add_argument('--stabilize', help='Stabilize image (default).', dest='stabilize', action='store_true')
parser.add_argument('--no-stabilize', help='Do not stabilize image.', dest='stabilize', action='store_false')
parser.set_defaults(stabilize=True)

parser.add_argument('--remove-shadows', help='Pixels that look like shadows should not be considered part of the extracted object.', dest='remove_shadows', action='store_true')
parser.add_argument('--no-remove-shadows', help='Pixels that look like shadows should be considered part of the extracted object (default).', dest='remove_shadows', action='store_false')
parser.set_defaults(remove_shadows=False)

args = vars(parser.parse_args())
print ("Args: %s" % args)


def main():
    cap = cv2.VideoCapture(args['source'])
    if cap is None or not cap.isOpened():
        print('Error: unable to open video source')
        return -1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args['height'])

    backgroundSubstractor = cv2.createBackgroundSubtractorMOG2()

    idle = True
    force_start = args['autostart']

    # A buffer holding the frames. It will hold up to args['buffer'] framesselfself.
    frames = None

    while(True):
        # Capture frame-by-frame.
        if not cap:
            break
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

        if ret:
            # Display the current frame
            if args['show']:
                cv2.imshow('frame', current)
                cv2.moveWindow('frame', 0, 0)

            if idle:
                # We are not capturing at the moment.
                print("Idle, proceeding")
                continue

            if len(frames) < args['buffer'] and cap.isOpened():
                # We are not done buffering.
                print("Got %d/%d frames" % (len(frames), args['buffer']))
                frames.append(current)
                continue

        # At this stage, we are done buffering, either because there are no more
        # frames at hand or because we have enough frames. Stop recording, start
        # processing.
        idle = True

        if args['autoexit']:
            cap.release()
            cap = None

        print("Capture complete.")

        if args['stabilize']:
            print("Stabilizing.")
            frames = stabilize(frames)

        # Extract foreground
        foreground = None
        for frame in frames:
            foreground = backgroundSubstractor.apply(frame) # FIXME: Is this the right subtraction?

        mask = foreground
        if args['remove_shadows']:
        # Smoothen a bit the mask to get back some of the missing pixels
            mask = cv2.bitwise_and(mask, 255)

        mask = cv2.blur(mask, (args['blur'], args['blur']))
        ret, mask = cv2.threshold(foreground, 1, 255, cv2.THRESH_BINARY)

        color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        if args['show']:
            cv2.imshow('tracking', mask)
            cv2.moveWindow('tracking', args['width'] + 32, args['height'] + 32)

        if args['dump_mask']:
            cv2.imwrite(args['dump_mask'], color_mask)

        tracking = cv2.bitwise_and(color_mask, frames[-2])
        if args['show']:
            cv2.imshow('objects', tracking)
            cv2.moveWindow('objects', 0, args['height'] + 32)

        if args['dump_object']:
            cv2.imwrite(args['dump_object'], tracking)

        if cap and not cap.isOpened():
            break

    # When everything done, release the capture
    if cap:
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

    raw_writer = None
    stabilized_writer = None

    if args['dump_raw']:
        raw_writer = cv2.VideoWriter(args['dump_raw'], cv2.VideoWriter_fourcc(*"DIVX"), 16, (args['width'], args['height']));
    if args['dump_stabilized']:
        stabilized_writer = cv2.VideoWriter(args['dump_stabilized'], cv2.VideoWriter_fourcc(*"DIVX"), 16, (args['width'], args['height']));

    # Stabilize image, most likely introducing borders.
    stabilized.append(prev)
    for cur in frames[1:]:
        if raw_writer:
            raw_writer.write(cur)
        cur_gray = cv2.cvtColor(cur, cv2.COLOR_RGB2GRAY)

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

                result = None

                if dx == 0. and dy == 0.:
                    print("stabilize: dx and dy are 0")
                    # For some reason I don't understand yet, if both dx and dy are 0,
                    # our matrix multiplication doesn't seem to make sense.
                    result = cur
                else:
                    da = math.atan2(transform[1, 0], transform[0, 0])

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
                    print("stabilize: resized full transform\n %s" % numpy.round(acc_transform[0:2, :]))
                    result = cv2.warpAffine(cur, numpy.round(acc_transform[0:2,:]), (args['width'], args['height']), cv2.INTER_NEAREST)
                stabilized.append(result)

                if stabilized_writer:
                    stabilized_writer.write(result)
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