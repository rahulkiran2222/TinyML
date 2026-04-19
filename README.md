# 🧠 TinyML Smart Motion Detector

### A Hands-On Workshop: Deploying Neural Networks on ESP32 Microcontrollers with TensorFlow Lite

[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow-Lite%20Micro-FF6F00?logo=tensorflow)](https://www.tensorflow.org/lite/microcontrollers)
[![ESP32](https://img.shields.io/badge/Platform-ESP32-E7352C?logo=espressif)](https://www.espressif.com/)
[![Arduino](https://img.shields.io/badge/IDE-Arduino-00979D?logo=arduino)](https://www.arduino.cc/)
[![Python](https://img.shields.io/badge/Training-Python%203.8+-3776AB?logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![Cost](https://img.shields.io/badge/Total%20Cost-%E2%82%B9800-blue)]()

> **Target Audience**: Undergraduate ECE / EEE / CSE students, embedded systems enthusiasts, and educators seeking a practical introduction to edge AI.

---

<p align="center">
  <img src="https://img.shields.io/badge/⚡_From_Sensor_→_Neural_Network_→_LED_in_Under_2_Hours-black?style=for-the-badge" />
</p>

---

## 📋 Table of Contents

1. [Introduction](#-introduction)
2. [What is TinyML?](#-what-is-tinyml)
3. [Why This Matters](#-why-this-matters-for-ece-students)
4. [Project Overview](#-project-overview)
5. [System Architecture](#-system-architecture)
6. [Bill of Materials](#-bill-of-materials)
7. [Hardware Setup](#-step-1-hardware-setup)
8. [Software Environment](#-step-2-software-environment)
9. [Data Collection](#-step-3-data-collection)
10. [Model Training](#-step-4-model-training)
11. [Deployment to ESP32](#-step-5-deployment-to-esp32)
12. [How It All Works Together](#-how-it-all-works-together)
13. [Learning Objectives](#-learning-objectives)
14. [Troubleshooting](#-troubleshooting)
15. [Going Further](#-going-further--challenge-problems)
16. [Recommended Reading](#-recommended-reading--references)
17. [Contributing](#-contributing)
18. [License](#-license)

---

## 📖 Introduction

This repository contains a **complete, beginner-friendly workshop** for building a **Smart Motion Detector** — a real-time embedded ML system that classifies physical motion as **"fast"** or **"slow"** using a neural network running *entirely* on a ₹600 microcontroller.

No cloud. No GPU. No internet connection required at inference time.

The workshop walks through the **full ML pipeline on the edge**:

```
Raw Sensor Data → Feature Extraction → Model Training → Quantization → MCU Deployment → Real-Time Inference
```

It was originally developed as a resource for **ECE undergraduates** exploring the intersection of embedded systems and machine learning. Whether you're a student, a professor building a lab exercise, or a hobbyist — this guide is for you.

---

## 🔬 What is TinyML?

**TinyML** is the field of deploying machine learning models on **ultra-low-power, resource-constrained devices** — typically microcontrollers (MCUs) with:

| Resource | Typical MCU | Typical PC/Server |
|----------|------------|-------------------|
| **RAM** | 256 KB – 1 MB | 8 – 64 GB |
| **Storage** | 1 – 4 MB Flash | 256 GB – 2 TB |
| **Power** | 1 – 500 mW | 65 – 300 W |
| **Clock Speed** | 80 – 240 MHz | 3 – 5 GHz |
| **Cost** | $2 – $10 | $500 – $5,000 |

### TensorFlow Lite for Microcontrollers

**TensorFlow Lite Micro (TFLM)** is Google's open-source framework that enables this. The workflow:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Train Model │     │  Convert to  │     │  Quantize    │     │  Deploy on   │
│  (TensorFlow │────▶│  TFLite      │────▶│  (INT8 /     │────▶│  MCU (C/C++) │
│   / Keras)   │     │  (.tflite)   │     │   Float16)   │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
    Python               Python              Python              Arduino/C++
    ~Minutes             ~Seconds            ~Seconds             Real-Time
```

### Why ESP32?

The **ESP32** (by Espressif Systems) is the ideal TinyML learning platform:

| Feature | ESP32 Specification | Why It Matters |
|---------|-------------------|----------------|
| **Processor** | Dual-core Xtensa LX6 @ 240 MHz | Fast enough for real-time inference |
| **SRAM** | 520 KB | Fits small neural networks in memory |
| **Flash** | 4 MB | Stores the model + firmware |
| **Wi-Fi** | 802.11 b/g/n | Optional: send results to a dashboard |
| **Bluetooth** | BLE 4.2 | Optional: connect to a phone app |
| **GPIO** | 34 pins | Connect sensors, LEDs, actuators |
| **Price** | ₹500 – ₹700 | Accessible for every student |
| **Ecosystem** | Arduino, PlatformIO, ESP-IDF | Massive community and documentation |

---

## 🎯 Why This Matters for ECE Students

### Industry Relevance

```
┌─────────────────────────────────────────────────────────────────┐
│                    TinyML Career Landscape                      │
├─────────────────┬──────────────────┬────────────────────────────┤
│  Embedded AI    │  IoT + ML        │  Edge Computing            │
│  Engineers      │  Solutions        │  Architects               │
├─────────────────┼──────────────────┼────────────────────────────┤
│  Qualcomm       │  Bosch            │  NVIDIA (Jetson)          │
│  Texas Instr.   │  Siemens          │  Google (Coral)           │
│  STMicro        │  TCS / Wipro      │  Amazon (Greengrass)      │
│  NXP            │  Startups         │  Microsoft (Azure Edge)   │
└─────────────────┴──────────────────┴────────────────────────────┘
```

### Skills You'll Develop

| Skill | Traditional ECE | This Workshop |
|-------|----------------|---------------|
| Microcontroller Programming | ✅ | ✅ |
| Sensor Interfacing | ✅ | ✅ |
| Signal Processing | ✅ | ✅ |
| Machine Learning | ❌ | ✅ |
| Model Optimization | ❌ | ✅ |
| Edge Deployment | ❌ | ✅ |
| Python + C/C++ Integration | ❌ | ✅ |

### Real-World Applications of TinyML

| Domain | Application | Sensor |
|--------|------------|--------|
| 🌾 Smart Agriculture | Soil health classification | Moisture, pH |
| 🏥 Healthcare | Cough / anomaly detection | Microphone |
| 🏭 Industry 4.0 | Predictive maintenance | Vibration (accelerometer) |
| 🏠 Smart Home | Gesture recognition | Ultrasonic, IR |
| 👶 Wearables | Fall detection | IMU (accelerometer + gyro) |
| 🔋 Energy | Power consumption anomaly | Current sensor |

---

## 🚀 Project Overview

### What We're Building

A **Smart Motion Detector** that:
1. **Senses** distance using an HC-SR04 ultrasonic sensor
2. **Computes** motion speed from consecutive readings
3. **Classifies** the motion as "fast" or "slow" using a trained neural network
4. **Indicates** the result with LEDs (🟢 Green = Slow, 🔴 Red = Fast)

### Conceptual Diagram

```
    ┌─────────┐        ┌─────────────┐        ┌──────────────┐
    │  Moving  │  sound │   HC-SR04   │ analog │    ESP32     │
    │  Object  │◄──────▶│  Ultrasonic │───────▶│ MCU + TFLite │
    │  (Hand)  │  waves │   Sensor    │  GPIO  │   Model      │
    └─────────┘        └─────────────┘        └──────┬───────┘
                                                      │ GPIO
                                               ┌──────▼───────┐
                                               │  🟢 Green LED │ ← Slow
                                               │  🔴 Red LED   │ ← Fast
                                               └──────────────┘
```

### The ML Pipeline at a Glance

| Phase | Tool | Output |
|-------|------|--------|
| **1. Data Collection** | Arduino IDE + Serial Monitor | `motion_data.csv` |
| **2. Model Training** | Python + TensorFlow (Colab) | `motion_model.tflite` |
| **3. Model Conversion** | `xxd` CLI tool | `model.h` (C byte array) |
| **4. Deployment** | Arduino IDE + TFLite ESP32 lib | Firmware on ESP32 |
| **5. Inference** | ESP32 (real-time) | LED output |

---

## ⚙️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ESP32 FIRMWARE                                   │
│                                                                          │
│  ┌──────────────┐    ┌───────────────┐    ┌───────────────────────────┐ │
│  │  HC-SR04      │    │   Feature     │    │   TFLite Micro            │ │
│  │  Driver       │───▶│   Extraction  │───▶│   Interpreter             │ │
│  │  (distance)   │    │   (speed calc)│    │   (neural network)        │ │
│  └──────────────┘    └───────────────┘    └─────────────┬─────────────┘ │
│                                                          │               │
│                                                  ┌───────▼────────┐     │
│                                                  │  Decision Logic │     │
│                                                  │  prediction     │     │
│                                                  │  < 0.5 → Slow   │     │
│                                                  │  ≥ 0.5 → Fast   │     │
│                                                  └───────┬────────┘     │
│                                                          │               │
│                                                  ┌───────▼────────┐     │
│                                                  │  GPIO Output   │     │
│                                                  │  🟢 Pin 13     │     │
│                                                  │  🔴 Pin 14     │     │
│                                                  └────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
```

### Neural Network Architecture

```
Input Layer          Hidden Layer 1       Hidden Layer 2       Output Layer
[distance] ──┐      ┌──[n1]──┐           ┌──[n1]──┐          ┌──────────┐
              ├──────┤  ...   ├───────────┤  ...   ├──────────┤ sigmoid  │──▶ P(fast)
[speed]    ──┘      └──[n8]──┘           └──[n4]──┘          └──────────┘
                    (8 neurons,           (4 neurons,          (1 neuron,
                     ReLU)                 ReLU)                Sigmoid)

Total Parameters: ~57
Model Size (Quantized): ~1–2 KB
Inference Time: <1 ms on ESP32
```

---

## 💰 Bill of Materials

| # | Component | Quantity | Approx. Cost (₹) | Purchase Link |
|---|-----------|----------|-------------------|---------------|
| 1 | ESP32 DevKit V1 | 1 | 500–700 | [Robu.in](https://robu.in) / Amazon.in |
| 2 | HC-SR04 Ultrasonic Sensor | 1 | 100–150 | [Robu.in](https://robu.in) |
| 3 | Green LED (5mm) | 1 | 5–10 | Any electronics store |
| 4 | Red LED (5mm) | 1 | 5–10 | Any electronics store |
| 5 | 220Ω Resistors | 2 | 5 | Any electronics store |
| 6 | Breadboard (400-point) | 1 | 80–100 | [Robu.in](https://robu.in) |
| 7 | Jumper Wires (M-M, M-F) | ~10 | 50–80 | Any electronics store |
| 8 | Micro-USB Cable | 1 | 50–100 | Often included with ESP32 |
| | | **Total** | **₹800 – ₹1,150** | |

> 💡 **Tip**: Most ECE labs already have breadboards, LEDs, and jumper wires. You may only need to purchase the ESP32 and HC-SR04.

---

## 🔧 Step 1: Hardware Setup

### Wiring Diagram

```
                    ┌──────────────────────┐
                    │       ESP32          │
                    │                      │
 HC-SR04           │   VIN (5V) ◄─────────┤◄── VCC (Red)
 Ultrasonic        │   GND     ◄─────────┤◄── GND (Black)
 Sensor            │   GPIO 5  ◄─────────┤◄── Trig (Yellow)
                    │   GPIO 18 ◄─────────┤◄── Echo (Orange)
                    │                      │
                    │   GPIO 13 ──────────┤──▶ 220Ω ──▶ 🟢 Green LED ──▶ GND
                    │   GPIO 14 ──────────┤──▶ 220Ω ──▶ 🔴 Red LED   ──▶ GND
                    │                      │
                    │        USB           │
                    └──────────┬───────────┘
                               │
                          To Laptop
```

### Pin Mapping Table

| Component | Component Pin | ESP32 Pin | Wire Color (Suggested) |
|-----------|--------------|-----------|----------------------|
| HC-SR04 | VCC | VIN (5V) | Red |
| HC-SR04 | GND | GND | Black |
| HC-SR04 | Trig | GPIO 5 | Yellow |
| HC-SR04 | Echo | GPIO 18 | Orange |
| Green LED | Anode (+) | GPIO 13 (via 220Ω) | Green |
| Green LED | Cathode (−) | GND | Black |
| Red LED | Anode (+) | GPIO 14 (via 220Ω) | Red |
| Red LED | Cathode (−) | GND | Black |

> ⚠️ **Important**: The HC-SR04 operates at 5V logic. The ESP32's GPIO pins are 3.3V tolerant but can read the 5V Echo signal in most cases. For a production design, use a voltage divider on the Echo pin. For this workshop, direct connection works fine.

---

## 💻 Step 2: Software Environment

### Required Software

| Tool | Purpose | Installation |
|------|---------|-------------|
| **Arduino IDE 2.x** | ESP32 firmware development | [arduino.cc/en/software](https://www.arduino.cc/en/software) |
| **Python 3.8+** | Model training | [python.org](https://www.python.org/) or use Google Colab |
| **Google Colab** (optional) | Cloud-based training (no GPU needed locally) | [colab.research.google.com](https://colab.research.google.com/) |

### Arduino IDE Setup

1. **Install Arduino IDE** from [arduino.cc](https://www.arduino.cc/en/software)

2. **Add ESP32 Board Support**:
   - Go to `File → Preferences`
   - In "Additional Board Manager URLs", add:
     ```
     https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
     ```
   - Go to `Tools → Board → Boards Manager`
   - Search for **"ESP32"** by Espressif Systems → **Install**

3. **Install TensorFlow Lite Library**:
   - Go to `Sketch → Include Library → Manage Libraries`
   - Search for **"TensorFlow Lite ESP32"** → **Install**

4. **Select Your Board**:
   - `Tools → Board → ESP32 Arduino → ESP32 Dev Module`
   - `Tools → Port → [Select your COM port]`

### Python Dependencies (for model training)

```bash
pip install tensorflow pandas numpy scikit-learn
```

Or simply use **Google Colab**, which has everything pre-installed.

---

## 📊 Step 3: Data Collection

We need labeled sensor data to train our classifier. The HC-SR04 measures distance to a moving object, and we compute speed from consecutive readings.

### Classification Criteria

| Label | Speed Threshold | Example Motion |
|-------|----------------|----------------|
| **Slow** (0) | < 5 cm/s | Slow hand wave, gradual approach |
| **Fast** (1) | ≥ 5 cm/s | Quick swipe, sudden movement |

### Data Collection Firmware

Upload this sketch to your ESP32:

```cpp
// data_collector.ino — Collects distance & speed data via Serial

#define TRIG_PIN 5
#define ECHO_PIN 18

void setup() {
    Serial.begin(115200);
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    Serial.println("distance,speed"); // CSV header
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

    // Measure echo time → distance
    duration = pulseIn(ECHO_PIN, HIGH);
    distance = duration * 0.034 / 2; // Speed of sound: 0.034 cm/μs

    // Calculate speed (change in distance / time interval)
    if (prev_distance != 0) {
        speed = abs(distance - prev_distance) / 0.1; // 100ms interval
        Serial.print(distance);
        Serial.print(",");
        Serial.println(speed);
    }
    prev_distance = distance;
    delay(100); // Sample every 100ms → 10 Hz
}
```

### Data Collection Protocol

| Step | Duration | Action | Label |
|------|----------|--------|-------|
| 1 | 30 seconds | Wave hand **slowly** in front of sensor (20–50 cm range) | `slow` |
| 2 | 30 seconds | Wave hand **quickly** in front of sensor | `fast` |
| 3 | 30 seconds | Mix of slow and fast movements | Both |
| 4 | — | Repeat until you have **100–200 rows** | — |

### Creating the CSV

1. Open **Serial Monitor** (115200 baud)
2. Perform the motions described above
3. Copy the output into a file called `motion_data.csv`
4. **Manually add the `label` column** based on which phase you were in:

```csv
distance,speed,label
10.5,2.3,slow
12.1,1.8,slow
15.2,8.9,fast
8.7,12.4,fast
11.0,3.1,slow
...
```

> 📝 **Note**: Aim for a **balanced dataset** — roughly equal numbers of "slow" and "fast" samples.

---

## 🧠 Step 4: Model Training

Use this code in **Google Colab** or a local Python environment.

### Training Script

```python
# train_model.py — Train and export a TinyML motion classifier

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ─── 1. LOAD DATA ───────────────────────────────────────────
data = pd.read_csv("motion_data.csv")
print(f"Dataset size: {len(data)} samples")
print(f"Class distribution:\n{data['label'].value_counts()}")

X = data[['distance', 'speed']].values
y = data['label'].map({'slow': 0, 'fast': 1}).values

# ─── 2. SPLIT DATA ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

# ─── 3. BUILD MODEL ─────────────────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── 4. TRAIN ────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

# ─── 5. EVALUATE ─────────────────────────────────────────────
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy:.2%}")
print(f"📉 Test Loss: {loss:.4f}")

# ─── 6. CONVERT TO TFLITE (QUANTIZED) ────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

with open("motion_model.tflite", "wb") as f:
    f.write(quantized_model)

print(f"\n📦 Model saved: motion_model.tflite")
print(f"📏 Model size: {len(quantized_model)} bytes ({len(quantized_model)/1024:.1f} KB)")
```

### Expected Output

```
Dataset size: 180 samples
Training: 144 | Testing: 36

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 8)                 24
 dense_1 (Dense)             (None, 4)                 36
 dense_2 (Dense)             (None, 1)                 5
=================================================================
Total params: 65
Trainable params: 65
Non-trainable params: 0
_________________________________________________________________

✅ Test Accuracy: 94.44%
📉 Test Loss: 0.1523
📦 Model saved: motion_model.tflite
📏 Model size: 1.2 KB
```

> 🎯 A 65-parameter model in 1.2 KB — that's TinyML in action!

---

## 📲 Step 5: Deployment to ESP32

### 5.1 Convert `.tflite` to C Header

The ESP32 can't read `.tflite` files directly. We embed the model as a **C byte array**.

**On Linux / macOS / WSL:**
```bash
xxd -i motion_model.tflite > model.h
```

**On Windows (PowerShell alternative):**
```powershell
$bytes = [System.IO.File]::ReadAllBytes("motion_model.tflite")
$hex = ($bytes | ForEach-Object { "0x{0:x2}" -f $_ }) -join ", "
$content = "const unsigned char motion_model_tflite[] = { $hex };`nconst unsigned int motion_model_tflite_len = $($bytes.Length);"
Set-Content -Path "model.h" -Value $content
```

### 5.2 Edit `model.h`

Open the generated file and ensure the array is declared as `const`:

```cpp
// model.h — Auto-generated TFLite model as C byte array
#ifndef MODEL_H
#define MODEL_H

const unsigned char motion_model_tflite[] = {
    0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33,
    // ... (remaining bytes)
};
const unsigned int motion_model_tflite_len = 1234;

#endif // MODEL_H
```

### 5.3 Inference Firmware

Place `model.h` in the same folder as this sketch, then upload:

```cpp
// smart_motion_detector.ino — TinyML inference on ESP32

#include <TensorFlowLite_ESP32.h>
#include "model.h"

#define TRIG_PIN   5
#define ECHO_PIN   18
#define GREEN_LED  13
#define RED_LED    14

// ─── TensorFlow Lite Micro Setup ────────────────────────────
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

tflite::MicroErrorReporter   micro_error;
tflite::AllOpsResolver        resolver;
const tflite::Model*          model;
tflite::MicroInterpreter*     interpreter;
TfLiteTensor*                 input;
TfLiteTensor*                 output;
uint8_t tensor_arena[8 * 1024]; // 8 KB memory arena

void setup() {
    Serial.begin(115200);
    pinMode(TRIG_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);
    pinMode(GREEN_LED, OUTPUT);
    pinMode(RED_LED, OUTPUT);

    // Load model from Flash
    model = tflite::GetModel(motion_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("❌ Model version mismatch!");
        while (1); // Halt
    }
    Serial.println("✅ Model loaded successfully");

    // Initialize interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, sizeof(tensor_arena), &micro_error
    );
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();

    input  = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("✅ Interpreter ready. Starting inference...\n");
}

void loop() {
    long duration;
    float distance, prev_distance = 0, speed;

    // ─── Read Sensor ────────────────────────────────────────
    digitalWrite(TRIG_PIN, LOW);
    delayMicroseconds(2);
    digitalWrite(TRIG_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIG_PIN, LOW);

    duration = pulseIn(ECHO_PIN, HIGH);
    distance = duration * 0.034 / 2;

    if (prev_distance != 0) {
        speed = abs(distance - prev_distance) / 0.1;

        // ─── Run Inference ──────────────────────────────────
        input->data.f[0] = distance;
        input->data.f[1] = speed;
        interpreter->Invoke();

        float prediction = output->data.f[0];

        // ─── Output Results ─────────────────────────────────
        Serial.printf("Distance: %.1f cm | Speed: %.1f cm/s | P(fast): %.3f → %s\n",
                       distance, speed, prediction,
                       prediction < 0.5 ? "🟢 SLOW" : "🔴 FAST");

        if (prediction < 0.5) {
            digitalWrite(GREEN_LED, HIGH);
            digitalWrite(RED_LED, LOW);
        } else {
            digitalWrite(GREEN_LED, LOW);
            digitalWrite(RED_LED, HIGH);
        }
    }
    prev_distance = distance;
    delay(100);
}
```

### 5.4 Upload & Test

1. Open the sketch in Arduino IDE
2. Ensure `model.h` is in the same directory
3. Select `ESP32 Dev Module` as the board
4. Click **Upload** ▶️
5. Open **Serial Monitor** (115200 baud)
6. Wave your hand — watch the LEDs and serial output!

### Expected Serial Output

```
✅ Model loaded successfully
✅ Interpreter ready. Starting inference...

Distance: 25.3 cm | Speed: 2.1 cm/s | P(fast): 0.087 → 🟢 SLOW
Distance: 24.8 cm | Speed: 5.0 cm/s | P(fast): 0.423 → 🟢 SLOW
Distance: 18.2 cm | Speed: 66.0 cm/s | P(fast): 0.971 → 🔴 FAST
Distance: 12.5 cm | Speed: 57.0 cm/s | P(fast): 0.945 → 🔴 FAST
Distance: 11.9 cm | Speed: 6.0 cm/s | P(fast): 0.312 → 🟢 SLOW
```

---

## 🔄 How It All Works Together

```
┌─────────────────────────────────────────────────────────────────────┐
│                        COMPLETE PIPELINE                            │
│                                                                     │
│  OFFLINE (Once)                    ONLINE (Real-Time, On Device)    │
│  ─────────────                    ──────────────────────────────    │
│                                                                     │
│  ┌──────────┐   ┌──────────┐     ┌──────────┐   ┌──────────────┐  │
│  │ Collect  │   │  Train   │     │ HC-SR04  │   │   ESP32       │  │
│  │ Data     │──▶│  Model   │     │ reads    │──▶│   computes    │  │
│  │ (CSV)    │   │ (Python) │     │ distance │   │   speed       │  │
│  └──────────┘   └────┬─────┘     └──────────┘   └──────┬───────┘  │
│                      │                                   │          │
│                 ┌────▼─────┐                      ┌──────▼───────┐  │
│                 │ Convert  │                      │  TFLite      │  │
│                 │ to .h    │─────────────────────▶│  Inference   │  │
│                 │ (xxd)    │    Embedded in       │  (<1ms)      │  │
│                 └──────────┘    firmware           └──────┬───────┘  │
│                                                          │          │
│                                                   ┌──────▼───────┐  │
│                                                   │  LED Output  │  │
│                                                   │  🟢 or 🔴     │  │
│                                                   └──────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

| Stage | What Happens | Where | Time |
|-------|-------------|-------|------|
| **Sensing** | Ultrasonic pulse measures distance to object | ESP32 + HC-SR04 | ~10ms |
| **Feature Extraction** | Speed calculated from consecutive distance readings | ESP32 (C++) | <1ms |
| **Inference** | Neural network predicts P(fast) from [distance, speed] | ESP32 (TFLite Micro) | <1ms |
| **Actuation** | LED activated based on prediction threshold (0.5) | ESP32 GPIO | Instant |
| **Total Latency** | End-to-end per frame | — | **~12ms** |

---

## 🎓 Learning Objectives

After completing this workshop, students will be able to:

| # | Objective | Bloom's Level |
|---|-----------|--------------|
| 1 | **Explain** the TinyML paradigm and its advantages over cloud-based ML | Understanding |
| 2 | **Interface** sensors with microcontrollers and collect structured data | Applying |
| 3 | **Train** a binary classification neural network using TensorFlow/Keras | Applying |
| 4 | **Convert** and **quantize** a Keras model to TensorFlow Lite format | Applying |
| 5 | **Deploy** a TFLite model on an ESP32 microcontroller | Applying |
| 6 | **Analyze** model performance and identify accuracy-latency tradeoffs | Analyzing |
| 7 | **Design** modifications to the system for new use cases | Creating |

---

## 🛠️ Troubleshooting

| Problem | Likely Cause | Solution |
|---------|-------------|----------|
| **Serial Monitor shows garbage** | Wrong baud rate | Set to **115200** |
| **"Model version mismatch!"** | TFLite library version mismatch | Update `TensorFlow Lite ESP32` library |
| **Sensor reads 0.0 cm always** | Wiring error | Check Trig→GPIO5, Echo→GPIO18 |
| **Sensor reads erratic values** | Object too close (<2cm) or too far (>400cm) | Keep hand 5–100 cm from sensor |
| **LEDs don't light up** | Resistor or polarity issue | Check anode (+) to GPIO, cathode (−) to GND |
| **Upload fails** | Wrong board selected | Ensure `ESP32 Dev Module` is selected |
| **Upload fails (timeout)** | Boot mode not entered | Hold **BOOT** button while uploading |
| **Model too large for arena** | `tensor_arena` too small | Increase from `8*1024` to `16*1024` |
| **Python training fails** | Missing dependencies | Run `pip install tensorflow pandas scikit-learn` |

---

## 🚀 Going Further — Challenge Problems

For students who finish early or want to explore deeper:

### 🟢 Beginner Extensions

| Challenge | Description |
|-----------|-------------|
| **Add a Buzzer** | Sound an alarm when fast motion is detected |
| **LCD Display** | Show speed and classification on a 16×2 LCD or OLED |
| **Data Logging** | Save predictions to an SD card with timestamps |

### 🟡 Intermediate Extensions

| Challenge | Description |
|-----------|-------------|
| **3-Class Classifier** | Add a "medium" speed category |
| **Wi-Fi Dashboard** | Send predictions to a web dashboard using ESP32's Wi-Fi |
| **Feature Engineering** | Add acceleration (change in speed) as a third input feature |
| **Cross-Validation** | Implement k-fold cross-validation in the training script |

### 🔴 Advanced Extensions

| Challenge | Description |
|-----------|-------------|
| **IMU-Based Gesture Recognition** | Replace ultrasonic with MPU6050 accelerometer; classify gestures |
| **Anomaly Detection** | Train an autoencoder to detect unusual motion patterns |
| **Federated Learning** | Multiple ESP32s contribute to a shared model without sharing raw data |
| **Power Profiling** | Measure actual power consumption during inference using a current sensor |
| **Compare with Cloud ML** | Send the same data to a cloud API and compare latency / accuracy |

---

## 📚 Recommended Reading & References

### Books

| Title | Authors | Why Read It |
|-------|---------|-------------|
| *TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers* | Pete Warden, Daniel Situnayake | **The** definitive TinyML textbook |
| *AI at the Edge* | Daniel Situnayake, Jenny Plunkett | Practical edge AI system design |

### Papers

1. **Warden, P., & Situnayake, D.** (2019). TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers. *O'Reilly Media.*

2. **Banbury, C. R., et al.** (2021). Benchmarking TinyML Systems: Challenges and Direction. *arXiv preprint arXiv:2003.04821.*

3. **David, R., et al.** (2021). TensorFlow Lite Micro: Embedded Machine Learning for TinyML Systems. *Proceedings of Machine Learning and Systems (MLSys).*

4. **Lin, J., et al.** (2020). MCUNet: Tiny Deep Learning on IoT Devices. *NeurIPS 2020.*

### Online Resources

| Resource | Link |
|----------|------|
| TensorFlow Lite Micro Documentation | [tensorflow.org/lite/microcontrollers](https://www.tensorflow.org/lite/microcontrollers) |
| ESP32 Arduino Core Documentation | [docs.espressif.com](https://docs.espressif.com/projects/arduino-esp32/) |
| TinyML Foundation | [tinyml.org](https://www.tinyml.org/) |
| Edge Impulse (No-Code TinyML) | [edgeimpulse.com](https://www.edgeimpulse.com/) |
| Harvard CS249r: Tiny Machine Learning | [tinyml.seas.harvard.edu](https://tinyml.seas.harvard.edu/) |

---

## 📁 Repository Structure

```
tinyml-smart-motion-detector/
│
├── README.md                       # ← You are here
├── LICENSE                         # MIT License
│
├── firmware/
│   ├── data_collector/
│   │   └── data_collector.ino      # Step 3: Data collection sketch
│   │
│   └── smart_motion_detector/
│       ├── smart_motion_detector.ino  # Step 5: Inference sketch
│       └── model.h                    # Converted TFLite model (generated)
│
├── training/
│   ├── train_model.py              # Step 4: Python training script
│   ├── motion_data.csv             # Sample dataset (or bring your own)
│   ├── motion_model.tflite         # Exported TFLite model (generated)
│   └── requirements.txt            # Python dependencies
│
├── docs/
│   ├── wiring_diagram.png          # Fritzing / hand-drawn wiring diagram
│   ├── pin_mapping.md              # Detailed pin reference
│   └── workshop_slides.pdf         # Presentation slides (if applicable)
│
└── assets/
    ├── demo.gif                    # Demo video / GIF
    └── architecture.png            # System architecture diagram
```

---

## 🤝 Contributing

Contributions are welcome! Whether you're fixing a typo, adding a new sensor example, or translating the workshop into another language.

1. **Fork** the repository
2. **Create** a branch: `git checkout -b feature/add-imu-example`
3. **Commit** with clear messages: `git commit -m "Add MPU6050 gesture recognition example"`
4. **Push** and open a **Pull Request**

### Contribution Ideas

- 🌐 **Translate** the README into Hindi, Tamil, Telugu, or other languages
- 📸 **Add photos** of your assembled hardware
- 📊 **Share your dataset** with labeled motion data
- 🧪 **Write unit tests** for the training pipeline
- 📹 **Record a video walkthrough** of the workshop

---

## 📄 License

This project is licensed under the **MIT License** — free to use, modify, and distribute for educational and commercial purposes.

```
MIT License · Copyright (c) 2024 TinyML Workshop Contributors
```

See [LICENSE](LICENSE) for the full text.

---

## 🙏 Acknowledgments

- **Google TensorFlow Team** — for TensorFlow Lite Micro
- **Espressif Systems** — for the incredible ESP32 platform
- **Pete Warden & Daniel Situnayake** — for pioneering TinyML education
- **Harvard CS249r** — for setting the standard in TinyML curriculum
- Our **professors and mentors** who encouraged us to explore the edge of AI — literally

---

<p align="center">
  <img src="https://img.shields.io/badge/Built_by_Rahul,_for_ECE_Students-🎓-blue?style=for-the-badge" />
</p>

<p align="center">
  <i>"The best way to predict the future is to build it — on a microcontroller."</i>
</p>

<p align="center">
  <b>⭐ If this workshop helped you learn something new, give it a star! ⭐</b>
</p>
