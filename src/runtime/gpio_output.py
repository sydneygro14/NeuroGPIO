# Sends prediction to Raspberry Pi GPIO pins.

# Sends prediction labels to Raspberry Pi GPIO pins.

from gpiozero import LED
from time import sleep


# Map each model label to the GPIO pin connected to that LED.
LEDS = {
    "left_hand": LED(17),
    "right_hand": LED(27),
    "feet": LED(22),
    "tongue": LED(23),
    "rest": LED(24),
}


def all_leds_off():
    # Turn off every LED.
    for led in LEDS.values():
        led.off()


def show_prediction(predicted_class: str):
    # Turn off old prediction before showing new one.
    all_leds_off()

    # If something unexpected comes in, show rest instead of crashing.
    if predicted_class not in LEDS:
        predicted_class = "rest"

    # Turn on the LED for the predicted class.
    LEDS[predicted_class].on()


def test_leds():
    # Test every LED one at a time.
    for class_name in LEDS:
        print("Testing:", class_name)
        show_prediction(class_name)
        sleep(1)

    all_leds_off()


if __name__ == "__main__":
    test_leds()