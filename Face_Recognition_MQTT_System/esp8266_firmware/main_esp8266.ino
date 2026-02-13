#include <ESP8266WiFi.h>
#include <PubSubClient.h>
#include <Servo.h>

// === WIFI & MQTT CONFIG ===
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* mqtt_server = "broker.hivemq.com";
const int mqtt_port = 1883;
const char* topic_command = "embedded_sys/face/command";

WiFiClient espClient;
PubSubClient client(espClient);
Servo panServo;

// Pin Definitions
const int SERVO_PIN = D4; // GPIO2 (built-in LED usually, check board specific)

int currentAngle = 90;

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
  
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.println(message);

  // Command Handling
  if (String(topic) == topic_command) {
    if (message == "L") {
      currentAngle -= 5;
      if (currentAngle < 0) currentAngle = 0;
      panServo.write(currentAngle);
      Serial.println("Moving Left");
    } else if (message == "R") {
      currentAngle += 5;
      if (currentAngle > 180) currentAngle = 180;
      panServo.write(currentAngle);
      Serial.println("Moving Right");
    } else if (message == "S") {
      // Stop logic (Servo simply holds position)
      Serial.println("Stop");
    }
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    String clientId = "ESP8266Client-";
    clientId += String(random(0xffff), HEX);
    
    if (client.connect(clientId.c_str())) {
      Serial.println("connected");
      client.subscribe(topic_command);
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
  setup_wifi();
  
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
  
  panServo.attach(SERVO_PIN);
  panServo.write(currentAngle); // Center position
}

void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();
}
