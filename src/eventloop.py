# pylint: disable=redefined-builtin,attribute-defined-outside-init
from __future__ import print_function

import time
from threading import Thread
from threading import Timer  # For running code after a delay
from Queue import Queue      # For a thread-safe event queue
import RPi.GPIO as GPIO      # So we can read Raspberry Pi GPIO pins

# This is onetime setup required by the GPIO module to specify that we
# want to refer to GPIO pins by their chipset number not the actual
# pin number on the circuit board. You can call setmode() yourself to override.
GPIO.setmode(GPIO.BCM)


# Python 2.7 uses raw_input while Python 3 deprecated it in favor of input.
# We want to use input() in either version
try:
    input = raw_input
except NameError:
    pass

#
# This EventLoop class enables event-based asynchronous programming where
# the events are GPIO button presses and timers. Typically, you create
# an EventLoop object, call monitorGPIOButton() one or more times to
# configure one or more buttons and then call the loop() method. The loop()
# method generally never returns: it blocks until an event occurs, then
# invokes the callback associated with that event, and then loops again
# waiting for another event.
#
# In addition to monitoring GPIO pins for events, you can also use later()
# to invoke functions after specified delays.
#
# Call exit() to force loop() to exit. In typical use, this will also make
# your entire program exit.
#
class EventLoop(object):
    def __init__(self):
        self.queue = Queue()
        self.debouncing = False

    # This method is a loop that removes a function from the queue
    # and invokes it. If the queue is empty, it blocks until something
    # is added. The loop runs forever or until the exit() method is called.
    def loop(self):
        self.running = True
        while self.running:
            f = self.queue.get()
            f()

    # This method causes the loop() method to exit.
    # If you called loop() as the last line of your program, this
    # will generally cause your program to exit as well.
    def exit(self):
        self.running = False         # Set the flag
        self.queue.put(lambda: None)  # Unblock the queue if necessary

    # This method waits the specified number of seconds and then puts
    # the function f onto the queue. Assuming that you have called loop()
    # this means that f() will be invoked as soon as possible after
    # delaySeconds has elapsed.
    def later(self, f, delaySeconds):
        t = Timer(delaySeconds, lambda: self.queue.put(f))
        t.start()
        return t

    #
    # Monitor a GPIO pin with a push button attached to it.
    #
    # When the pin state changes, this method puts a function on the
    # event queue to call the callback function with three arguments:
    #  - the pin number
    #  - the pin value: a 1 if the pin is now high or a zero if low
    #  - the time when the state change occurred
    #
    # By default, the button is expected to be wired between the pin and ground
    # and the pin will use the built-in pull-up resistor. In this default case
    # the pin will read high when the button is not pressed and will read
    # low when the button is pressed. If you wire your button to +3.3v instead
    # of ground, then pass pull_up=False to configure the pin to pull down.
    # to low when the switch is not pressed.
    #
    # By default, we wait 1ms after an edge is detected before reading the
    # pin value, to allow time to "debounce". Different buttons have
    # different amounts of bounce, however, so set debounce_time to a larger
    # value if your button requires it.
    #
    def monitor_gpio_pin(self, pin, callback,
                         pull_up=True,
                         debounce_time=.001):
        # Configure the pin
        GPIO.setup(pin, GPIO.IN,
                   pull_up_down=GPIO.PUD_UP if pull_up else GPIO.PUD_DOWN)

        # This function will get called when the pin changes from low to
        # high or high to low.
        def edge_handler(pin):
            # Something has happened on the pin, but it might still
            # be bouncing, so we set a timer and wait to check the pin state
            # until the timer expires. If there is already a timer running
            # then we just ignore this edge event completely.
            if self.debouncing:
                return

            # This is the function that gets called by the timer
            def debounce_handler():
                # Read the state of the pin: a 0 or 1
                state = GPIO.input(pin)
                # Put a function in the queue to call the callback
                self.queue.put(lambda: callback(pin, state))
                self.debouncing = False

            self.debouncing = True   # Ignore events on this pin while true
            # Start the debounce timer
            Timer(debounce_time, debounce_handler).start()

        # This is how we get edge_handler() called when the pin changes state
        GPIO.add_event_detect(pin, GPIO.BOTH, edge_handler)

    #
    # This is a higher-level version of monitorGPIOPin() that makes a
    # GPIO push button behave something like a GUI pushbutton. It can
    # detect clicks, double clicks and long presses. The callback
    # function is invoked whenever an event occurs. The first argument
    # to the callback is a string that specifies the event, and the
    # second argument is the pin number.
    #
    # If the user just clicks the button, the callback will be called
    # three times with arguments "press", "release" and "click". There
    # will be a short delay between the "release" event and the
    # "click" event to make sure that the click is not just the start
    # of a double-click. (Specify the doubleclick_speed argument to
    # control the length of this delay.)
    #
    # If the user double-clicks the button quickly enough, the
    # callback will be called five times: "press", "release", "press",
    # "doubleclick", "release"
    #
    # If the user presses and holds for long enough, the callback will
    # be called three times: "press", "longpress", "release". Specify
    # the longpress_duration argument to control how long the user must
    # hold the button down to trigger a longpress event.
    #
    # Most callers can ignore callback invocations where the first
    # argument is "press" or "release". Typically they will only want
    # to respond to the higher-level "click", "doubleclick" and
    # "longpress" events.
    #
    # See monitor_gpio_pin() for details on the pull_up and debounce_time
    # arguments.
    #
    def monitor_gpio_button(self, pin, callback,
                            pull_up=True,
                            debounce_time=.002,
                            longpress_duration=1,
                            doubleclick_speed=.1):

        # Create an object that will hold the state values shared by the
        # nested functions below. (In Python 3 we could use nonlocal variables
        # but for Python 2, we need to use object attributes.)
        class State(object):
            pass
        state = State()

        # We handle events using a simple finite state machine.
        # This is the state variable for that FSM.
        state.buttonstate = 0
        # We also store some timer objects so we can cancel them as needed
        state.longpress_timer = None
        state.doubleclick_timer = None

        def pin_handler(pin, pinstate):
            # Convert the 0/1 state of the pin to a pressed/released boolean
            pressed = not bool(pinstate) if pull_up else bool(pinstate)

            # Now handle this pressed or released event depending on
            # the current FSM state
            if state.buttonstate == 0:
                # this is the ground state when we have not received any events
                if pressed:
                    callback('press', pin)
                    state.longpress_timer = self.later(longpress_handler,
                                                       longpress_duration)
                    state.buttonstate = 1
                else:
                    # We can get a release in this state after a long press
                    # or after a doubleclick. For a normal button click we'll
                    # fire the release event in state 1
                    callback('release', pin)

            elif state.buttonstate == 1:
                # This is the state where the user has pressed the button
                if not pressed:
                    state.longpress_timer.cancel()
                    state.longpress_timer = None
                    callback('release', pin)
                    state.doubleclick_timer = self.later(doubleclick_handler,
                                                         doubleclick_speed)
                    state.buttonstate = 2
                else:
                    # We shouldn't get a press event here.
                    # But ignore it if we do
                    pass

            elif state.buttonstate == 2:
                # This is the state where we've gotten a press and release
                # and we're waiting for another press to start a doubleclick
                if pressed:
                    state.doubleclick_timer.cancel()
                    state.doubleclick_timer = None
                    callback('press', pin)
                    # For simplicity and responsiveness we fire the doubleclick
                    # event on the second press, without waiting for the release
                    # This means we don't need another state to wait for that
                    # release and we can just return to the ground state.
                    callback('doubleclick', pin)
                    state.buttonstate = 0
                else:
                    pass

        def longpress_handler():
            callback('longpress', pin)
            state.buttonstate = 0
            state.longpress_timer = None

        def doubleclick_handler():
            # This timer fires when it has been too long for
            # a doubleclick to start. At this point we can
            # safely fire the click event knowing that we're not
            # half-way through a double click
            callback('click', pin)
            state.buttonstate = 0
            state.doubleclick_timer = None

        # Monitor the specified pin, and call the pin_handler function
        # when something happens on it.
        self.monitor_gpio_pin(pin, pin_handler, pull_up, debounce_time)

    def monitor_console(self, callback, prompt='>'):
        def input_thread():
            while True:
                s = input(prompt)
                self.queue.put(lambda: callback(s))
                time.sleep(0.5)
        t = Thread(target=input_thread)
        t.daemon = True
        t.start()


# Some test code to demonstrate usage of this module
# if __name__ == "__main__":
#
#     loop = EventLoop()
#
#     # demonstrate the loop.later() method
#     def one():
#         print('one second')
#         loop.later(two, 1)
#     def two():
#         print('two seconds')
#         loop.later(three, 1)
#     def three():
#         print('three seconds')
#     loop.later(one, 1)
#
#     # Demonstrate the loop.monitor_gpio_button() method
#     # This assumes you have a push button wired between GPIO26 and ground
#     print('doubleclick three times to quit')
#     global doubleclicks
#     doubleclicks = 0
#
#     def button_handler(event, pin):
#         print(event)
#         global doubleclicks
#         if event == 'doubleclick':
#             doubleclicks += 1
#             if doubleclicks == 3:
#                 loop.exit()
#
#     loop.monitor_gpio_button(26, button_handler)
#
#     # Run the event loop forever or until loop.exit() is called
#     loop.loop()
