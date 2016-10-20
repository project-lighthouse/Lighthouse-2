import math
import sys

import cv2
import numpy

def init(args):
    # Does nothing for the moment.
    0

def acquire(args):
    """
    Acquire objects from a video source (camera or movie).
    acquire(argv) -> [images]
    """
    cap = cv2.VideoCapture(args['video_source'])
    if cap is None or not cap.isOpened():
        print('Error: unable to open video source')
        return -1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args['video_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args['video_height'])

    backgroundSubstractor = cv2.createBackgroundSubtractorKNN()

    idle = True
    force_start = args['autostart']

    # A buffer holding the frames. It will hold up to args['buffer'] framesselfself.
    frames = None

    # The size of the largest suffix of `frames` composed solely of stable frames.
    consecutive_stable_frames = 0
    surface = args['video_width'] * args['video_height']

    raw_writer = None
    if args['dump_raw_video']:
        raw_writer = cv2.VideoWriter(args['dump_raw_video'], cv2.VideoWriter_fourcc(*"DIVX"), 16, (args['video_width'], args['video_height']));

    while(True):
        # Capture frame-by-frame.
        if not cap:
            break
        ret, current = cap.read()

        key = None
        if args['show']:
            key = cv2.waitKey(1) & 0xFF
            # <q> or <Esc>: quit
            if key == 27 or key == ord('q'):
                break
            # <spacebar> or `force_start`: start detecting.
            elif key == ord(' '):
                force_start = True

        if force_start:
            force_start = False
            idle = False
            frames = []
            consecutive_stable_frames = 0

        if not ret:
            print("No more frames.")
            break

        # Display the current frame
        if args['show']:
            cv2.imshow('Capturing image.', current)
            cv2.moveWindow('Capturing image.', 0, 0)

        if raw_writer:
            raw_writer.write(current)

        if idle:
            # We are not capturing at the moment.
            print("Idle, proceeding.")
            continue

        if not cap.isOpened():
            print("Video source closed.")
            break

        if len(frames) > 0 and args['buffer_stable_frames'] > 0:
            diff = cv2.norm(frames[-1], current)
            print("Diff: %d <? %d" % (diff, surface * args['buffer_stability']))
            if diff <= surface * args['buffer_stability']:
                consecutive_stable_frames += 1
            else:
                consecutive_stable_frames = 0

        # We are not done buffering.
        if args['verbose']:
            print("Got %d/%d frames, %d/%d stable frames." % (len(frames), args['buffer'], consecutive_stable_frames, args['buffer_stable_frames']))
        frames.append(current)

        if len(frames) >= args['buffer']:
            if consecutive_stable_frames >= args['buffer_stable_frames']:
                # We have enough frames and enough stable frames.
                if args['verbose']:
                    print("We have enough stable frames.")
                break
            else:
                # Make way for more frames.
                frames.pop(0)
        if args['verbose']:
            print("Continuing acquisition.")

    print("Video acquisition complete.")

    # At this stage, we are done buffering, either because there are no more
    # frames at hand or because we have enough frames. Stop recording, start
    # processing.
    cap.release()
    cap = None


    if args['stabilize']:
        print("Stabilizing video.")
        frames = stabilize(frames)
        print("Video stabilization complete.")

    # Extract foreground
    candidates = []

    print("Acquiring moving object.")
    for i, frame in enumerate(frames):
        height, width = frame.shape[:2]

        mask = backgroundSubstractor.apply(frame) # FIXME: Is this the right subtraction?
        original_mask = mask.copy()

        if args['remove_shadows']:
            mask = cv2.bitwise_and(mask, 255)

        # Smoothen a bit the mask to get back some of the missing pixels
        if args['acquisition-blur'] > 0:
            mask = cv2.blur(mask, (args['acquisition-blur'], args['acquisition-blur']))

        ret, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        corners = [[0, 0], [height - 1, 0], [0, width - 1], [height - 1, width - 1]]

        score = cv2.countNonZero(mask)
        if args['fill_holes'] and score != surface:
            # Attempt to fill any holes.
            # At this stage, often, we have a mask surrounded by black and containing holes.
            # (this is not always the case -  sometimes, the mask is a cloud of points).
            positive = mask.copy()
            fill_mask = numpy.zeros((height + 2, width + 2), numpy.uint8)
            found = False
            for y,x in corners:
                if positive[y, x] == 0:
                    cv2.floodFill(positive, fill_mask, (x, y), 255)
                    found = True
                    break

            if found:
                filled = cv2.bitwise_or(mask, cv2.bitwise_not(positive))

                # Check if we haven't filled too many things, in which case
                # our fill operation actually decreased the quality of the
                # image.
                filled_score = cv2.countNonZero(filled)
                if filled_score < surface * .9:
                    has_empty_corners = False
                    for y, x in corners:
                        if filled[y, x] == 0:
                            has_empty_corners = True
                            break
                    if has_empty_corners:
                        # Apparently, we have managed to remove holes, without filling
                        # the entire frame.
                        score = filled_score
                        mask = filled
                        print("Improved to a score of %d" % score)

        bw_mask = mask

        if args['show']:
            cv2.imshow('mask', mask)
            cv2.moveWindow('mask', args['video_width'] + 32, args['video_height'] + 32)

        if args['use_contour']:
            image, contours, hierarchy = cv2.findContours(bw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # FIXME: We could remove small contours (here or later)
            bw_mask = numpy.zeros((height, width), numpy.uint8)
            for cnt in contours:
                if cv2.contourArea(cnt) > args['acquisition_min_size'] or 0:
                    hull = cv2.convexHull(cnt)
                    cv2.fillPoly(bw_mask, [hull], 255, 8)

            if args['contours_prefix']:
                dest = "%s_%d.png" % (args['contours_prefix'], i)
                print("Writing contours to %s." % dest)
                cv2.imwrite(dest, bw_mask)

        score = cv2.countNonZero(bw_mask)
        mask = cv2.cvtColor(bw_mask, cv2.COLOR_GRAY2RGB)
        extracted = cv2.bitwise_and(mask, frame)
        if args['show']:
            cv2.imshow('extracted', extracted)
            cv2.moveWindow('extracted', 0, args['video_height'] + 32)

        if score != surface:
            # We have captured the entire image. Definitely not a good thing to do.
            if i > len(frames) * args['buffer_init'] or i + 1 == len(frames):
                # We are done buffering
                candidates.append((score, mask, bw_mask, original_mask, extracted, i, 0))

        latest_score = score

    candidates.sort(key=lambda tuple: tuple[0], reverse=True)
    candidates = candidates[:args['acquisition-keep-objects']]

    results = []

    for candidate_index, candidate in enumerate(candidates):
        best_score, best_mask, best_bw_mask, best_original_mask, best_extracted, best_index, best_perimeter = candidate

        print ("Best score %d/%s" % (best_score, best_perimeter))

# Get rid of small components
        if args['acquisition_min_size'] > 0:
            number, components = cv2.connectedComponents(best_bw_mask)
            flattened = components.flatten()
            stats = numpy.bincount(flattened)
            # FIXME: Optimize this
            removing = 0
            for i, stat in enumerate(stats):
                if stat == 0:
                    continue
                if stat < args['acquisition_min_size']:
                    kill_list = components == i
                    best_mask[kill_list] = 0
                    best_extracted[kill_list] = 0
                    removing += 1

# Add transparency
        split_1, split_2, split_3 = cv2.split(best_extracted)
        transparency = cv2.merge([split_1, split_2, split_3, best_bw_mask])
        results.append(transparency)
        if args['objects_prefix']:
            dest = "%s_%d.png" % (args['objects_prefix'], candidate_index)
            print("Writing object to %s." % dest)
            cv2.imwrite(dest, transparency)
        if args['masks_prefix']:
            dest = "%s_%d.png" % (args['masks_prefix'], candidate_index)
            print("Writing mask to %s." % dest)
            cv2.imwrite(dest, best_original_mask)

    cv2.destroyAllWindows()
    results


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

    stabilized_writer = None

    if args['dump_stabilized']:
        stabilized_writer = cv2.VideoWriter(args['dump_stabilized'], cv2.VideoWriter_fourcc(*"DIVX"), 16, (args['video_width'], args['video_height']));

    # Stabilize image, most likely introducing borders.
    stabilized.append(prev)
    for cur in frames[1:]:
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
                    result = cv2.warpAffine(cur, numpy.round(acc_transform[0:2,:]), (args['video_width'], args['video_height']), cv2.INTER_NEAREST)
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

