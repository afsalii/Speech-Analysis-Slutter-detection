# 🎙️ Speech Analysis: Pause & Repetition Detection

## 📌 Project Overview

This project is an **automated speech analysis pipeline** developed in **Python** to detect disfluencies in audio recordings. It was built as part of a technical assignment for the **Zlaqa Intern Program**.

The system processes raw speech audio files and identifies:

* **Pause Segments** → Silent regions where the speaker hesitates
* **Repetitions** → Stuttered patterns such as:

  * Syllable-level (e.g., *"ba-ba-ball"*)
  * Word-level (e.g., *"I-I-I want"*)

---

## ⚙️ Technical Features

### 🔊 Audio Preprocessing

* **Peak Normalization**
  Ensures consistent volume across different audio inputs for reliable analysis

* **Pre-emphasis Filtering**
  Enhances high-frequency components to improve speech clarity and feature extraction

---

### 📊 Feature Extraction

* **RMS Energy**
  Tracks signal power over time to detect silence

* **MFCCs (Mel-Frequency Cepstral Coefficients)**
  Captures spectral features of speech for acoustic similarity comparison

---

### 🔍 Detection Pipeline

* **Energy-based Pause Detection**
  Uses configurable thresholds to detect silence segments

* **Acoustic Similarity Repetition Detection**
  Applies **Cosine Similarity** on MFCC features to identify repeated speech patterns

---

## 🧰 Tech Stack

* **Language:** Python 3.x
* **Libraries:**

  * `librosa` → Audio processing & feature extraction
  * `scipy` → Signal filtering
  * `numpy` → Numerical computations
  * `scikit-learn` → Cosine similarity
  * `SpeechRecognition` → Speech-to-text transcription

---

## 🚀 Installation & Usage

### 1. Clone the repository

```bash
git clone (https://github.com/afsalii/Speech-Analysis-Slutter-detection.git)
cd Speech-Analysis-Slutter-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the analysis

```bash
python main.py
```

---

## 🧠 Detection Logic

### 1. Pause Detection (Task 1)

The system detects pauses using **RMS Energy analysis**:

* **Thresholding:**
  A segment is marked as a pause if energy remains below **0.01**

* **Duration Filter:**
  Only segments longer than **0.6 seconds** are considered pauses
  → Avoids detecting natural speech gaps

---

### 2. Repetition Detection (Task 2)

A hybrid **acoustic + NLP approach** is used:

#### 🔹 Acoustic Comparison

* Sliding window over MFCC features
* Cosine Similarity > **0.985** → repetition detected

#### 🔹 Jump Mechanism

* Prevents over-counting
* Skips **1.5 seconds forward** after detection

#### 🔹 Pattern Reconstruction

* Maps detected timestamps to transcript
* Reconstructs stutter patterns like:

  * `"ba-ba-ball"`

---

## ⚠️ Challenges Faced

### 1. Over-counting Repetitions

* Issue: Sliding window detected the same repetition thousands of times
* Solution: Implemented **Jump Logic (cooldown mechanism)**

---

### 2. ASR Auto-Correction

* Problem: Speech-to-text models remove stutters
* Solution:

  * Detect using **acoustic similarity**
  * Map back to transcript for reconstruction

---

### 3. Background Noise Interference

* Issue: Noise prevented RMS energy from reaching zero
* Solution:

  * Applied **pre-emphasis filtering**
  * Fine-tuned energy thresholds

---

## 📤 Expected Output

```plaintext
File: sample.wav

Pause Segments:
[0.4s - 1.1s], [2.3s - 2.9s]

Total Pause Duration: 1.3s

Repetitions:
Detected pattern: "ba-ba-ball"
Repetition Count: 2
```

---

## 📈 Future Improvements

* Real-time speech processing
* Deep learning-based disfluency detection
* Better noise robustness
* Visualization dashboard for speech patterns

---

## 🤝 Contribution

Feel free to fork this repository and submit pull requests for improvements.

---

## 📄 License

This project is open-source and available under the MIT License.
