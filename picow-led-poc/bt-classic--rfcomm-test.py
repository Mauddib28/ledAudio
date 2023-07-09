import bluetooth as bt

print("Scanning Bluetooth Devices....")

devices = bt.discover_devices(lookup_names=True)

for addr, name in devices:
    print("%s : %s" % (name, addr))

dev_name = input("Enter device name: ")

dev = ""
check = False

for addr, name in devices:
    if dev_name == name:
        dev = addr
        check = True

if not check:
    print("Device Name Invalid!")
else:
    print("Sending data to %s : %s" % (dev_name, dev))

hostMACAddress = dev
port = 1
backlog = 1
size = 8
s = bt.BluetoothSocket(bt.RFCOMM)
try:
    s.connect((hostMACAddress, port))
except:
    print("Couldn't Connect!")
s.send("T")
s.send("E")
s.send("S")
s.send("T")
s.send(".")
s.close()