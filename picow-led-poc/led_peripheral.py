# PiCockpit.com
#	- URL:		https://picockpit.com/raspberry-pi/everything-about-bluetooth-on-the-raspberry-pi-pico-w/
## Note: This code functions an an RX/TX channel between two devices

###
# Edited by:		Paul Wortman	-	2023/07/09
###

import bluetooth
import random
import struct
import time
from machine import Pin
from ble_advertising import advertising_payload

from micropython import const

# Debugging
dbg = 0

_IRQ_CENTRAL_CONNECT = const(1)
_IRQ_CENTRAL_DISCONNECT = const(2)
_IRQ_GATTS_WRITE = const(3)

_FLAG_READ = const(0x0002)
_FLAG_WRITE_NO_RESPONSE = const(0x0004)
_FLAG_WRITE = const(0x0008)
_FLAG_NOTIFY = const(0x0010)

_UART_UUID = bluetooth.UUID("6E400001-B5A3-F393-E0A9-E50E24DCCA9E")
_UART_TX = (
    bluetooth.UUID("6E400003-B5A3-F393-E0A9-E50E24DCCA9E"),
    _FLAG_READ | _FLAG_NOTIFY,
)
_UART_RX = (
    bluetooth.UUID("6E400002-B5A3-F393-E0A9-E50E24DCCA9E"),
    _FLAG_WRITE | _FLAG_WRITE_NO_RESPONSE,
)
_UART_SERVICE = (
    _UART_UUID,
    (_UART_TX, _UART_RX),
)

# Class definition for the BLESimplePeripheral Object
class BLESimplePeripheral:
    def __init__(self, ble, name="mpy-uart"):
        # Sets the BLE object to the Class' internal property
        self._ble = ble
        # Sets the BLE radio to being on
        self._ble.active(True)
        # Registers a callback for events from the BLE stack; using the Class' _irq function at the BLE Object's callback
        self._ble.irq(self._irq)
        # Configures the server with the specified services; which replaces any existing services
        ((self._handle_tx, self._handle_rx),) = self._ble.gatts_register_services((_UART_SERVICE,))
        # Other configuration
        self._connections = set()
        self._write_callback = None
        self._payload = advertising_payload(name=name, services=[_UART_UUID])
        self._advertise()

    def _irq(self, event, data):
        # A central device has connected to this peripheral
        if event == _IRQ_CENTRAL_CONNECT:
            conn_handle, _, _ = data
            #print("New connection", conn_handle)
            print("[+] New connection\t-\t[ {0} ]".format(conn_handle))
            self._connections.add(conn_handle)
        # A central device has disconnected to this peripheral
        elif event == _IRQ_CENTRAL_DISCONNECT:
            conn_handle, _, _ = data
            #print("Disconnected", conn_handle)
            print("[-] Disconnected\t-\t[ {0} ]".format(conn_handle))
            self._connections.remove(conn_handle)
            self._advertise()
        # A client has written to this Characteristic or Descriptor
        elif event == _IRQ_GATTS_WRITE:
            conn_handle, value_handle = data
            value = self._ble.gatts_read(value_handle)
            if value_handle == self._handle_rx and self._write_callback:
                self._write_callback(value)
        '''
        # A single scan result; NOTE: This event is not defined
        elif event == _IRQ_SCAN_RESULT:
            print("[*] Single Scan Result:")
            addr_type, addr, adv_type, rssi, adv_data = data
            print("\tAddress Type:\t{0}\n\tAddress:\t\t{1}\n\tAdv Type:\t{2}\n\tRSSI:\t\t{3}\n\tAdv Data:\t\t{4}".format(addr_type, addr, adv_type, rssi, adv_data))
        # Scan duration finished or was manually stopped
        elif event == _IRQ_SCAN_DONE:
            print("[*] IRQ Scan Completed OR Stopped")
            pass
        '''

    def send(self, data):
        # Iterate through each connected device
        for conn_handle in self._connections:
            # Sends a notification request to the connected client
            self._ble.gatts_notify(conn_handle, self._handle_tx, data)
            # Getting odd error around here, might be part of additions above

    def is_connected(self):
        return len(self._connections) > 0

    # Internal function for advertising the BLE GAP Service
    def _advertise(self, interval_us=500000):
        print("[*] Starting advertising")
        # Call to advertise using the provided payload-data
        self._ble.gap_advertise(interval_us, adv_data=self._payload)
        # Note: adv_data is included in all broadcasts; where resp_data is sent in reply to an active scan

    # Internal function for having a callback after a write occurs
    def on_write(self, callback):
        self._write_callback = callback


def demo():
    led_onboard = Pin("LED", Pin.OUT)
    # Creating a BLE bluetooth device
    ble = bluetooth.BLE()
    # Passing the BLE device to the BLESimplePeripheral Class
    p = BLESimplePeripheral(ble)

    # Sub-function for use as callback function
    def on_rx(v):
        #print("RX", v)
        print("[+] RX:\t\t{0}".format(v))

    # Setting the callback function for the on_write() function
    p.on_write(on_rx)

    # While loop for tracking timing on TX checking
    i = 0
    while True:
        # Only perform the TX tracking if a device is connected to the Class Object
        if p.is_connected():
            # Turn on the LED on the Pico W
            led_onboard.on()
            for _ in range(3):
                data = str(i) + "_"
                #print("TX", data)
                if dbg != 0:
                    print("[*] TX:\t\t{0}".format(data))
                p.send(data)
                i += 1
        # Timing sleep (1/10th of a second)
        time.sleep_ms(100)


if __name__ == "__main__":
    print("[*] Beginning Demo")
    demo()
    print("[+] Completed Demo")