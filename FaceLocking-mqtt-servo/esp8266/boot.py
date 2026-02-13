# boot.py - run on boot-up
import network
import time
from machine import Pin

# Status LED (Onboard LED usually on Pin 2, inverted logic often)
led = Pin(2, Pin.OUT)
led.value(1) # Off

def connect_wifi(ssid, password):
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print('Connecting to network...')
        wlan.connect(ssid, password)
        while not wlan.isconnected():
            led.value(0) # Blink
            time.sleep(0.1)
            led.value(1)
            time.sleep(0.1)
            pass
    print('Network config:', wlan.ifconfig())
    led.value(0) # On (Connected)

# Configure your WiFi here
SSID = "YOUR_WIFI_SSID" 
PASSWORD = "YOUR_WIFI_PASSWORD"

try:
    connect_wifi(SSID, PASSWORD)
except Exception as e:
    print("WiFi Connection Failed:", e)
