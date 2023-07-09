from picozero import LED
from time import sleep

red_led = LED(22)
green_led = LED(29)
blue_led = LED(16)

blue_led.on()
sleep(5)
blue_led.off()