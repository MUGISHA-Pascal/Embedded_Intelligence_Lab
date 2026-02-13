from machine import Pin, PWM
from umqtt.simple import MQTTClient
import time
import ujson

# Configuration
# VPS IP for MQTT Broker
MQTT_BROKER = "10.12.73.9"  # Local PC IP for testing
CLIENT_ID = "esp8266_team313"
TOPIC_SUB = b"vision/team313/movement"
TOPIC_PUB = b"vision/team313/heartbeat"

# Servo Configuration
# SG90 Servo: 50Hz PWM. Duty cycle usually 40-115 for 0-180 degrees.
SERVO_PIN = 14 # D5 on NodeMCU
servo = PWM(Pin(SERVO_PIN), freq=50)

# Positions (Duty values for 0, 90, 180 degrees approx)
# 10-bit resolution (0-1023)
# Min (0 deg) ~ 0.5ms/20ms * 1023 = ~26
# Max (180 deg) ~ 2.4ms/20ms * 1023 = ~123
# Center (90 deg) ~ 1.45ms/20ms * 1023 = ~74
MIN_DUTY = 40
MAX_DUTY = 115
CENTER_DUTY = 77

current_duty = CENTER_DUTY

def set_servo(duty):
    global current_duty
    # Clamp
    if duty < MIN_DUTY: duty = MIN_DUTY
    if duty > MAX_DUTY: duty = MAX_DUTY
    servo.duty(duty)
    current_duty = duty
    print("Servo Duty:", duty)

def sub_cb(topic, msg):
    global current_duty
    print((topic, msg))
    
    try:
        data = ujson.loads(msg)
        status = data.get("status", "")
        
        step = 5 # Move 5 units per command
        
        if status == "MOVE_LEFT":
            # If camera needs to move left, servo moves... direction depends on mounting.
            # Assuming increasing duty moves left.
            set_servo(current_duty + step)
        elif status == "MOVE_RIGHT":
            set_servo(current_duty - step)
        elif status == "CENTERED":
            # Do nothing? Or strict center? 
            # Assignment says "camera rotates based on face movement".
            # Usually we just hold position if centered.
            pass
            
    except Exception as e:
        print("Error parsing JSON:", e)

def main():
    print("Starting MQTT Client...")
    
    # Initialize Servo to Center
    set_servo(CENTER_DUTY)
    
    try:
        client = MQTTClient(CLIENT_ID, MQTT_BROKER)
        client.set_callback(sub_cb)
        client.connect()
        print("Connected to MQTT Broker:", MQTT_BROKER)
        client.subscribe(TOPIC_SUB)
        print("Subscribed to:", TOPIC_SUB)
        
        last_heartbeat = 0
        
        while True:
            # Check for messages
            client.check_msg()
            
            # Heartbeat every 10s
            now = time.time()
            if now - last_heartbeat > 10:
                payload = ujson.dumps({"node": "esp8266", "status": "ONLINE", "uptime": now})
                client.publish(TOPIC_PUB, payload)
                last_heartbeat = now
                
            time.sleep(0.1)
            
    except Exception as e:
        print("Error:", e)
        print("Rebooting in 5 seconds...")
        time.sleep(5)
        import machine
        machine.reset()

if __name__ == "__main__":
    main()
