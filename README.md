# TinyML
What is TinyML and TensorFlow Lite?
TinyML is about running lightweight machine learning models on resource-constrained devices like microcontrollers (MCUs). Imagine adding intelligence to a soil moisture sensor or a motion detector without a Wi-Fi connection—that’s TinyML! TensorFlow Lite is Google’s framework to make this happen. It takes big ML models (like neural networks), shrinks them (via techniques like quantization), and runs them on tiny hardware with just kilobytes of RAM and milliwatts of power.
The ESP32 is our star MCU here. It’s a dual-core, Wi-Fi-enabled chip with 520 KB SRAM and 4 MB Flash—way more capable than an Arduino Uno, yet cheap and perfect for TinyML. You’ve probably used it in IoT projects; now, let’s make it think!

Why This Matters for ECE Students
Hardware + Software Fusion: Combines your microcontroller programming (C/C++) with ML—skills companies like TCS, Wipro, or startups crave.
Real-Time Applications: From smart agriculture to wearables, TinyML is everywhere.
Affordable: ESP32 and sensors cost less than ₹1,000 total—ideal for Indian students.

Project: "Smart Motion Detector"
For this workshop, we’ll build a Smart Motion Detector using the ESP32 and an HC-SR04 ultrasonic sensor. It’ll classify movements as “fast” or “slow” in real-time using a TinyML model, lighting an LED (green for slow, red for fast). Think of it as a mini security system or a gesture-controlled switch—simple yet powerful!
Step-by-Step Guide
1. Hardware Setup
What You’ll Need:
ESP32 DevKit V1 (₹500–₹700 on Amazon India or Robu.in).
HC-SR04 Ultrasonic Sensor (₹100–₹150) – measures distance to detect motion.
LEDs: Green and Red (₹10 each) + 220Ω resistors.
Breadboard and Jumper Wires (₹100–₹200).
USB Cable: For power and programming.
Connections:
HC-SR04:
VCC → 5V (ESP32’s VIN pin).
GND → GND.
Trig → GPIO 5.
Echo → GPIO 18.
LEDs:
Green: Anode → GPIO 13 (via resistor), Cathode → GND.
Red: Anode → GPIO 14 (via resistor), Cathode → GND.
Plug the ESP32 into your laptop via USB. Done? Let’s move to software!
2. Software Tools
Arduino IDE: Free, install it from arduino.cc. Add ESP32 support via Boards Manager (search “ESP32” by Espressif).
Python: For training the model (use Google Colab if your PC is slow).
TensorFlow Lite: We’ll convert a model to run on the ESP32.
ESP32 TensorFlow Lite Library: Install it in Arduino IDE (Library Manager → “TensorFlow Lite ESP32”).
3. Collecting Data
We need data to train our model. The HC-SR04 measures distance (in cm) to a moving object. We’ll classify motion speed:
Slow: Object moves < 5 cm/second.
Fast: Object moves > 5 cm/second.
Code to Collect Data (Upload this to ESP32 via Arduino IDE):
cpp
#define TRIG_PIN 5
#define ECHO_PIN 18

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
}

void loop() {
  long duration;
  float distance, prev_distance = 0, speed;

  // Trigger ultrasonic pulse
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Measure echo time
  duration = pulseIn(ECHO_PIN, HIGH);
  distance = duration * 0.034 / 2; // Distance in cm

  // Calculate speed (cm/s)
  if (prev_distance != 0) {
    speed = abs(distance - prev_distance) / 0.1; // 100ms delay
    Serial.print(distance);
    Serial.print(",");
    Serial.println(speed);
  }
  prev_distance = distance;
  delay(100); // Sample every 100ms
}
Open Serial Monitor (115200 baud).
Wave your hand in front of the sensor: slowly for 30 seconds, then quickly for 30 seconds.
Copy the output (e.g., “10.5,2.3” → distance, speed) into a CSV file:
distance,speed,label
10.5,2.3,slow
15.2,8.9,fast
Aim for 100–200 rows total.
4. Training the Model (Python/Google Colab)
We’ll train a small neural network using TensorFlow. Use this code in a Colab notebook:
python
import tensorflow as tf
import pandas as pd
import numpy as np

# Load your CSV data (upload to Colab)
data = pd.read_csv("motion_data.csv")
X = data[['distance', 'speed']].values
y = data['label'].map({'slow': 0, 'fast': 1}).values

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Quantize for ESP32 (optional, reduces size)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Save model
with open("motion_model.tflite", "wb") as f:
    f.write(quantized_model)
Download motion_model.tflite to your PC.
5. Deploying to ESP32
Now, convert the .tflite file to a C array for Arduino:
Use this command in your terminal (Linux/Mac) or WSL (Windows):
bash
xxd -i motion_model.tflite > model.h
Open model.h, rename the array to motion_model_tflite and add const:
cpp
const unsigned char motion_model_tflite[] = { ... };
const unsigned int motion_model_tflite_len = ...;
Here’s the full ESP32 code to run the model:
cpp
#include <TensorFlowLite_ESP32.h>
#include "model.h" // Your converted model

#define TRIG_PIN 5
#define ECHO_PIN 18
#define GREEN_LED 13
#define RED_LED 14

// TensorFlow Lite setup
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

tflite::MicroErrorReporter micro_error;
tflite::AllOpsResolver resolver;
const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
uint8_t tensor_arena[8 * 1024]; // 8KB arena (adjust if needed)

void setup() {
  Serial.begin(115200);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);

  // Load model
  model = tflite::GetModel(motion_model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch!");
    return;
  }

  // Initialize interpreter
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, sizeof(tensor_arena), &micro_error);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  input = interpreter->input(0);
  output = interpreter->output(0);
}

void loop() {
  long duration;
  float distance, prev_distance = 0, speed;

  // Get sensor data
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);
  duration = pulseIn(ECHO_PIN, HIGH);
  distance = duration * 0.034 / 2;

  if (prev_distance != 0) {
    speed = abs(distance - prev_distance) / 0.1;

    // Feed data to model
    input->data.f[0] = distance;
    input->data.f[1] = speed;
    interpreter->Invoke();

    // Get prediction
    float prediction = output->data.f[0];
    Serial.print("Prediction: ");
    Serial.println(prediction);

    // Control LEDs
    if (prediction < 0.5) { // Slow
      digitalWrite(GREEN_LED, HIGH);
      digitalWrite(RED_LED, LOW);
    } else { // Fast
      digitalWrite(GREEN_LED, LOW);
      digitalWrite(RED_LED, HIGH);
    }
  }
  prev_distance = distance;
  delay(100);
}
Include model.h in your sketch folder.
Upload to ESP32 and test by moving your hand!
6. How It Works
Sensor: HC-SR04 measures distance every 100ms.
Speed Calculation: Difference in distance over time.
Model: Predicts “slow” (0) or “fast” (1) based on distance and speed.
Output: Green LED for slow, red for fast.
