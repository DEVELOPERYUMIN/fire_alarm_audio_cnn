# 🔥 Fire Alarm Audio Classification CNN

> **CNN-based classification of fire alarm sounds and other repetitive / periodic audio events**  
>  
> This project focuses on detecting fire alarm sounds using a Log-Mel Spectrogram + CNN pipeline,  
> while being **generalizable to various repetitive and periodic sound classification tasks**.

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

  <figure style="text-align: center;">
    <img
      src="https://github.com/user-attachments/assets/396203c8-0cd4-4d3d-a98b-35e682485980"
      width="420"
    />
    <figcaption><b>Training Loss Curve</b></figcaption>
  </figure>

  <figure style="text-align: center;">
    <img
      src="https://github.com/user-attachments/assets/8fa5d814-e15f-4337-b331-e235826900f7"
      width="420"
    />
    <figcaption><b>Confusion Matrix</b></figcaption>
  </figure>

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
