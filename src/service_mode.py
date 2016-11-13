"""Main file for the device service mode."""
from __future__ import print_function

import argparse
import logging.config
import subprocess

from eventloop import EventLoop

MAX_SPEAKER_VOLUME = 50
MAX_MICROPHONE_VOLUME = 95

# The EventLoop object
eventloop = None
# Keep track of whether we're currently busy or not
busy = False
# Keeps current configuration values.
config = {}

parser = argparse.ArgumentParser(description="Lighthouse Service mode")
parser.add_argument('--gpio-pin',
                    help='What GPIO pin the button is attached to', default=26,
                    type=int)
parser.add_argument('--config-path', help='Path to the config file.',
                    required=True)
parser.add_argument('--audio-setup-path',
                    help='Path to the audio setup script to test changes.',
                    required=True)
args = parser.parse_args()

# Setup logging.
logging.config.dictConfig({
    'version': 1,
    'formatters': {
        'standard': {
            'format': '%(module)s:%(levelname)s %(message)s'
        }
    },
    'handlers': {
        'default': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        }
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
})
logger = logging.getLogger(__name__)


def say(sentence):
    subprocess.call(["espeak", "-s", "150", str(sentence)])
    logger.info(sentence)


def volume_section_activated():
    say("Short press to increase volume. Longpress to return to the main menu.")


def change_speaker_volume():
    volume = int(config['SPEAKER_VOLUME'])

    # Increase volume with "5" step or reset to minimum if maximum is reached.
    config['SPEAKER_VOLUME'] = 5 if volume >= MAX_SPEAKER_VOLUME else \
        volume + 5

    save_config()

    say(config['SPEAKER_VOLUME'])


def change_microphone_volume():
    volume = int(config['MICROPHONE_VOLUME'])

    # Increase volume with "5" step or reset to minimum if maximum is reached.
    config['MICROPHONE_VOLUME'] = 5 if volume >= MAX_MICROPHONE_VOLUME else \
        volume + 5

    save_config()

    say(config['MICROPHONE_VOLUME'])


def exit_to_main_menu():
    global current_section
    say("You are in the main menu now.")
    current_section = menu


def reboot_device():
    say("Device will be rebooted now.")
    subprocess.call(["shutdown", "-r", "now"])


def navigate_main_menu():
    # Increase menu item index or start from the beginning.
    current_section['index'] = 0 if \
        current_section['index'] == len(current_section['items']) - 1 else \
        current_section['index'] + 1

    # Say menu item name aloud.
    say(current_section['items'][current_section['index']]['name'])


def activate_menu_item():
    global current_section

    menu_item = current_section['items'][current_section['index']]

    if 'activate' in menu_item:
        current_section = menu_item
        menu_item['activate']()


menu = {
    "click": navigate_main_menu,
    "longpress": activate_menu_item,
    "index": -1,
    "items": [{
        "name": "Change speaker volume.",
        "activate": volume_section_activated,
        "click": change_speaker_volume,
        "longpress": exit_to_main_menu
    }, {
        "name": "Change microphone volume.",
        "activate": volume_section_activated,
        "click": change_microphone_volume,
        "longpress": exit_to_main_menu
    }, {
        "name": "Reboot device",
        "activate": reboot_device
    }]
}
current_section = menu


def save_config():
    # Save config values in format KEY=VALUE with dedicated line for every key.
    with open(args.config_path, "w") as config_file:
        for k, v in config.items():
            config_file.write("{}={}\n".format(k, v))

    # Apply volume settings.
    subprocess.call(["bash", args.audio_setup_path])


def ready():
    global busy
    busy = False


def button_handler(event, pin):
    # If we're still processing some other event, ignore the new one.
    if busy:
        logger.debug("Ignoring event %s.", event)
        return

    logger.debug("Pin #%s is activated by '%s' event.", pin, event)

    # If current section has handler for the current event - delegate event
    # processing to it.
    if event in current_section:
        current_section[event]()
        eventloop.later(ready, 0.5)


def main():
    # Monitor the button for events.
    global eventloop
    eventloop = EventLoop()
    eventloop.monitor_gpio_button(args.gpio_pin, button_handler,
                                  doubleclick_speed=0)

    say("Device is in Service Mode. Press button to navigate through the menu. "
        "Long press to enter the menu section or perform an action.")

    # Read current configuration values.
    with open(args.config_path) as config_file:
        for line in config_file:
            key, value = line.partition("=")[::2]
            key = key.strip()
            # Ignore empty lines.
            if not key:
                config[key] = value.strip()

    logger.info('Current environment variables: %s', config)

    ready()

    # Run the event loop forever.
    eventloop.loop()

if __name__ == '__main__':
    main()
