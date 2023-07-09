#!/usr/bin/python3

# Import for Raspi GPIO    -   For Pico-W
from machine import Pin, PWM
# Import for time
from time import sleep

# Configuration of the GPIO -   For Pico-W
red_led = Pin(17, mode=Pin.OUT)
green_led = Pin(22, mode=Pin.OUT)
blue_led = Pin(16, mode=Pin.OUT)

# Main Code

# Turn on the lights
print("[*] Turning on each LED; one at a time")
red_led.on()
print("[+] Red LED\t-\tON")
green_led.on()
print("[+] Green LED\t-\tON")
blue_led.on()
print("[+] Blue LED\t-\tON")
print("[+] All LEDs are ON")

sleep(3)

print("[*] Turning off each LED; one at a time")
red_led.off()
print("[+] Red LED\t-\tOFF")
green_led.off()
print("[+] Green LED\t-\tOFF")
blue_led.off()
print("[+] Blue LED\t-\tOFF")
print("[+] All LEDs are OFF")

sleep(2)

# Testing PWM
pwm__red_led = PWM(red_led)
pwm__green_led = PWM(green_led)
pwm__blue_led = PWM(blue_led)
# Set the freq for the PWM
pwm_freq = 1000
pwm__red_led.freq(pwm_freq)
pwm__green_led.freq(pwm_freq)
pwm__blue_led.freq(pwm_freq)

# Loops for Testing
print("[*] Stepping LEDs up and down; all together")
# Moving LEDs brighter
for duty in range(0, 65025):
    pwm__red_led.duty_u16(duty)
    pwm__green_led.duty_u16(duty)
    pwm__blue_led.duty_u16(duty)
    sleep(0.0001)
print("[+] Done moving up; switching to down")
sleep(1)
for duty in range(65025, 0, -1):
    pwm__red_led.duty_u16(duty)
    pwm__green_led.duty_u16(duty)
    pwm__blue_led.duty_u16(duty)
    sleep(0.0001)
sleep(1)
print("[+] Done moving down")
print("[+] Completed PWM test")


# End of Code
print("[+] Completed LED test on Raspi Pico-W")
