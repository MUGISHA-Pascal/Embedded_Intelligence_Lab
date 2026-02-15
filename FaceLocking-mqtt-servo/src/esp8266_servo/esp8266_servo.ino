/*
 * ESP8266 Servo Control via MQTT
 * Part of Phase 1: Distributed Vision-Control System
 * 
 * Dependencies:
 * - PubSubClient (by Nick O'Leary)
 * - ESP8266WiFi
 * - Servo
 */

#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>
#include <ArduinoJson.h>

// WiFi Configuration - UPDATE THESE
const char* ssid = "EdNet";
const char* password = "Huawei@123";

// MQTT Configuration
const char* mqtt_server = "157.173.101.159";
const char* mqtt_topic_servo = "robotics/team355/servo";
const char* mqtt_topic_events = "robotics/team355/events";

WiFiClient espClient;
PubSubClient client(espClient);
Servo myServo;

const int servoPin = D5;  // GPIO2

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);

  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("");
  Serial.println("WiFi connected");
  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("] ");

  String message;
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.println(message);

  // Parse JSON
  DynamicJsonDocument doc(256);
  DeserializationError error = deserializeJson(doc, message);

if (error) {
    Serial.print("deserializeJson() failed: ");
    Serial.println(error.c_str());     // or error.f_str() — both usually work
    return;
}

  if (String(topic) == mqtt_topic_servo) {
    if (doc.containsKey("angle")) {
      int angle = doc["angle"];
      angle = constrain(angle, 0, 180);
      myServo.write(angle);
      Serial.print("Moving servo to: ");
      Serial.println(angle);
    }
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    // Create a random client ID
    String clientId = "ESP8266Client-Team355-";
    clientId += String(random(0xffff), HEX);

    if (client.connect(clientId.c_str())) {
      Serial.println("connected");
      client.subscribe(mqtt_topic_servo);
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" try again in 5 seconds");
      delay(5000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  myServo.attach(servoPin);
  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}
