import cv2
from eventloop import EventLoop
from image_database import ImageDatabase
import audioutils
import time
import json
import os

# Load configuration from a config file rather than parsing a bunch of
# command-line arguments
with open('config.json', 'r') as f:
    config = json.load(f)

# Figure out where we're storing our data
config['db_root_dir'] = os.path.abspath(os.path.expanduser(config['db_root_dir']))

print('Configuration')
print(config)


# Set up the audio devices if they are configured
if config['audio_out_device']:
    audioutils.ALSA_SPEAKER = config['audio_out_device']
if config['audio_in_device']:
    audioutils.ALSA_MICROPHONE = config['audio_out_device']

# Define some sounds that we will be playing
start_recording_tone = audioutils.makebeep(800, .2)
stop_recording_tone = audioutils.makebeep(400, .2)

with open('sounds/shutter.raw', 'rb') as f:
    shutter = f.read()

# This is the central repository of items we know about
db = ImageDatabase(config['db_root_dir'])

print([item.dirname for item in db.items])

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
    # XXX Shouldn't assume that we'll always get a match.
    # this function should be defined so that it returns an array of
    # zero or more matching items.
    item = db.match(image)
    audioutils.playfile(item.audio_filename())

def record_new_item():
    image = take_picture(config)
    audioutils.play(start_recording_tone)
    audio = audioutils.record()
    audioutils.play(stop_recording_tone)
    item = db.add(image, audio)
    print("added image in {}".format(item.dirname))

def button_handler(event, pin):
    if event == 'click':
        match_item()
    elif event == 'longpress':
        record_new_item()

eventloop = EventLoop()
eventloop.monitor_gpio_button(26, button_handler);
eventloop.loop()

