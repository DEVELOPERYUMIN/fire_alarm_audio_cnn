# 🔥 Fire Alarm Audio Classification CNN

> **CNN-based classification of fire alarm sounds and other repetitive / periodic audio events**  
>  
> This project focuses on detecting fire alarm sounds using a Log-Mel Spectrogram + CNN pipeline,  
> while being **generalizable to various repetitive and periodic sound classification tasks**.

---

## 🏢 Indoor Fire Evacuation Autonomous Drone System – Context

### 🎯 System Goal

This CNN model is a core component of an **Indoor Fire Evacuation Autonomous Drone System**,  
whose primary goal is to **detect fire-related emergency situations early and guide occupants to safe evacuation routes in real time**.

In indoor environments where visibility is limited and GPS is unavailable, **sound-based fire detection becomes a critical sensing modality**.  
This system leverages **audio perception, autonomous drones, and intelligent decision-making** to support rapid and reliable evacuation guidance during fire emergencies.

---

### 🧩 Overall System Architecture

The full system is designed as a **multi-modal, drone-assisted disaster response platform**, consisting of the following components:

#### 🔊 Audio-Based Fire Detection (This CNN Model)
- Real-time microphone input (on drone or edge device)
- Fire alarm / emergency sound detection using CNN

#### 🚁 Autonomous Indoor Drone Platform
- Indoor navigation using **LiDAR + SLAM**
- Obstacle avoidance and path planning (**A\***, **D\* Lite**)
- Autonomous movement toward detected emergency zones

#### 🧍 Human Detection & Localization
- Thermal camera–based human detection
- Bluetooth / BLE-based proximity sensing
- Multi-sensor fusion for robust localization

#### 🧭 Evacuation Guidance System
- Visual / auditory / haptic feedback
- Rope-guided evacuation or directional signaling
- Mobile app integration for user interaction

#### 🖥 Backend & Monitoring
- **FastAPI-based server**
- Real-time status visualization
- Emergency event logging and decision support

---

### 🔊 Role of This CNN Model in the System

This **Fire Alarm Audio Classification CNN** is used at the **very first stage of the disaster response pipeline**.

Its role is to:

- Continuously monitor ambient audio
- Detect fire alarm sounds or other emergency-like repetitive signals
- Trigger the activation of the autonomous evacuation system

Once a fire alarm sound is detected with high confidence:

1. The detection event is sent to the control system
2. The autonomous drone is dispatched toward the detected area
3. Human detection and evacuation guidance modules are activated
4. Real-time evacuation assistance begins

➡️ **In short:**

> This CNN model acts as the **auditory trigger** that initiates the entire autonomous fire evacuation process.

---

## 📌 Project Overview

This repository contains a **Convolutional Neural Network (CNN)**–based audio classification model designed to detect **fire alarm sounds** from audio input.

Rather than relying on semantic understanding of sound, the model learns **time–frequency patterns** that commonly appear in alarm-like audio signals, such as:

- Strong periodicity  
- Repetitive frequency structures  
- High-energy spectral bands  

Because of this design, the model is not limited to fire alarms and can be **easily adapted to classify other repetitive or structured sound events**.

---

## 🎯 Core Idea

> **Repetitive sounds exhibit consistent patterns in the time–frequency domain.**

This project leverages the fact that alarm sounds, warning beeps, sirens, and similar audio events form **stable and repeatable spectrogram patterns**.

Pipeline:
1. Raw audio waveform
2. Log-Mel Spectrogram transformation
3. CNN-based spatial pattern learning
4. Sound event classification

The model focuses on **pattern recognition**, not on domain-specific assumptions.

<img
  src="https://github.com/user-attachments/assets/9a3915a8-16c1-480f-a33f-b516d630763e"
  alt="Training Results"
  width="600"
/>


---

## 🧠 Model Architecture

- **Input Representation**
  - Log-Mel Spectrogram
  - Sampling Rate: `22050 Hz`
  - `n_mels = 64`
  - `n_fft = 2048`
  - `hop_length = 512`

- **Model**
  - CNN-based audio classification network
  - Compatible with lightweight backbones (e.g., MobileNetV2, EfficientNet-Lite)

- **Output**
  - Class probability for sound events

> The audio classification task is treated as an image classification problem in the time–frequency domain.

---

## 📦 Pretrained Model

The pretrained MobileNetV2-based model  
**`mobilenetv2_050_exp001.pt`**  
is provided and can be used directly for inference.


 <div style="display: flex; justify-content: center; gap: 40px; align-items: flex-start;">
  <img
    src="https://github.com/user-attachments/assets/396203c8-0cd4-4d3d-a98b-35e682485980"
    alt="Training Loss Curve"
    width="420"
  />
  <img
    src="https://github.com/user-attachments/assets/8fa5d814-e15f-4337-b331-e235826900f7"
    alt="Confusion Matrix"
    width="420"
  />
</div>


---

## 🔄 Generalization Capability

Although trained primarily on **fire alarm sounds**, this model can be reused for other tasks with minimal modification.

### Applicable Sound Types
- Emergency sirens and alarms
- Industrial warning sounds
- Machine alert beeps
- Periodic mechanical noises
- Structured environmental audio events

### Why It Generalizes Well
- Learns **periodic spectral patterns**
- Robust to background noise
- Focuses on frequency repetition rather than absolute sound identity
- Easily retrainable with new labels and datasets

➡️ Simply replace the dataset and retrain to adapt the model to a new repetitive sound classification task.

---

## 🚀 Potential Applications

- 🔔 Fire and emergency alarm detection
- 🏭 Industrial anomaly and alert monitoring
- 🤖 Robotics and drone-based disaster response
- 🏢 Smart building sound monitoring
- 📱 Real-time microphone-based event detection systems

---

## 🧪 Training & Evaluation

- Log-Mel Spectrogram preprocessing
- Optional data augmentation
  - Time shifting
  - Noise injection
- Cross-validation supported
- Real-time inference pipeline available

---

## 📈 Future Work

- [ ] Validation on diverse repetitive sound datasets
- [ ] Multi-class and multi-label sound event detection
- [ ] Real-time streaming optimization
- [ ] Edge device deployment (Jetson, mobile)
- [ ] Integration with embedded or robotic systems


---

## 👤 Author

**Yumin Ahn**  
AI Audio Classification · CNN · Edge & Embedded AI  
Disaster Response & Intelligent Systems Research
