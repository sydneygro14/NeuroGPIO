# Simulated GPIO output for running NeuroGPIO on a Mac. (no pi connection yet)
#
# This file does not control real Raspberry Pi pins.
# It only prints which LED would turn on.

from time import sleep


def all_leds_off():
    # In simulation mode, there are no real LEDs to turn off.
    print("[SIM] all LEDs off")


def show_prediction(predicted_class: str):
    # Print which PCB LED would turn on.
    print(f"[SIM] LED ON for prediction: {predicted_class}")


def test_leds():
    # Test each simulated LED.
    classes = ["left_hand", "right_hand", "feet", "tongue", "rest"]

    for class_name in classes:
        show_prediction(class_name)
        sleep(1)

    all_leds_off()


if __name__ == "__main__":
    test_leds()