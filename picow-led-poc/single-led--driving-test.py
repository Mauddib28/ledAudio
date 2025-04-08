#!/usr/bin/python3

# Import for Raspi GPIO    -   For Pico-W
from machine import Pin
# Import for time
from time import sleep

# Configuration of the GPIO -   For Pico-W
red_led = Pin(22, mode=Pin.OUT)
green_led = Pin(29, mode=Pin.OUT)
blue_led = Pin(16, mode=Pin.OUT)

# Main Code

# Turn on the lights
print("[*] Turning on each LED; one at a time")
red_led.value(255)
print("[+] Red LED\t-\tON")
green_led.value(255)
print("[+] Green LED\t-\tON")
blue_led.value(255)
print("[+] Blue LED\t-\tON")
print("[+] All LEDs are ON")

sleep(2)

print("[*] Turning off each LED; one at a time")
red_led.value(0)
print("[+] Red LED\t-\tOFF")
green_led.value(0)
print("[+] Green LED\t-\tOFF")
blue_led.value(0)
print("[+] Blue LED\t-\tOFF")
print("[+] All LEDs are OFF")

sleep(2)

print("[*] Stepping LEDs up and down; all together")
# Moving LEDs brighter
for light_value in range(0, 255):
    red_led.value(light_value)
    green_led.value(light_value)
    blue_led.value(light_value)
print("[+] Done moving up; switching to down")
# Moving LEDs dimmer
for light_value in range(255, 0):
    red_led.value(light_value)
    green_led.value(light_value)
    blue_led.value(light_value)
print("[+] Done moving down")

# End of Code
print("[+] Completed LED test on Raspi Pico-W")
