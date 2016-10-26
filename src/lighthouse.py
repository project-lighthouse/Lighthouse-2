import cv2
import capture.capture as capture
from matching.matcher import Matcher
from eventloop import EventLoop
import audioutils
import time
import json

# Load configuration from a config file rather than parsing a bunch of
# command-line arguments
with open('config.json', 'r') as f:
    config = json.load(f)
print('Configuration')
print(config)

# Define some sounds that we will be playing
start_recording_tone = audioutils.makebeep(800, .2)
stop_recording_tone = audioutils.makebeep(400, .2)

with open('sounds/shutter.raw', 'rb') as f:
    shutter = f.read()

# XXX: we can probably optimize this by opening the camera
# on button press before we get the click or longpress...
# also, releasing the camera takes some time, too, so do that
# in another thread, or when we're competely done?
def take_picture(config):
    camera = cv2.VideoCapture(config['video_source'])

    if camera is None or not camera.isOpened():
        print('Error: unable to open camera')
        return -1

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config['video_width'])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config['video_height'])
    camera.set(cv2.CAP_PROP_FPS, config['video_fps'])

    retval, frame = camera.read()
    audioutils.play(shutter);

    camera.release()

    return frame if retval else None

def match_item():
    image = take_picture(config)
    print(len(image))

def record_new_item():
    print('record')

def button_handler(event, pin):
    if event == 'click':
        match_item()
    elif event == 'longpress':
        record_new_item()

eventloop = EventLoop()
eventloop.monitor_gpio_button(26, button_handler);
eventloop.loop()

