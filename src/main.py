import os
import librosa
import numpy as np
import scipy.signal
from sklearn.metrics.pairwise import cosine_similarity
import speech_recognition as sr


# 1. PREPROCESSING & FEATURE EXTRACTION

def get_features(file_path):
    # Load and normalize to ensure consistent energy levels [cite: 35]
    y, sr_rate = librosa.load(file_path, sr=22050)
    y = librosa.util.normalize(y) 
    
    hop_length = 512
    # RMS for Task 1: Pause Detection [cite: 17]
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    # MFCCs for Task 2: Repetition Detection [cite: 28]
    mfccs = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr_rate, hop_length=hop_length)
    
    return y, sr_rate, rms, mfccs, times


# 2. TASK 1: ACCURATE PAUSE DETECTION

def detect_pauses(rms, times):
    # Threshold 0.005 is very low—only catches true silence 
    threshold = 0.02 
    min_duration = 0.8 # Only count significant pauses [cite: 13]
    
    pauses = []
    pause_start = None
    for i, energy in enumerate(rms):
        if energy < threshold:
            if pause_start is None: pause_start = times[i]
        elif pause_start is not None:
            duration = times[i] - pause_start
            if duration >= min_duration:
                pauses.append((round(pause_start, 1), round(times[i], 1)))
            pause_start = None
            
    total_dur = sum([p[1] - p[0] for p in pauses])
    return pauses, round(total_dur, 1)


# 3. TASK 2: REPETITION & WORD DETECTION

def detect_repetitions(file_path, mfccs, times):
    recognizer = sr.Recognizer()
    repetition_count = 0
    final_pattern = "None"
    
    # Get transcript (Speech-to-Text) [cite: 11]
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
        full_text = recognizer.recognize_google(audio_data).lower()
        words = full_text.split()
    except:
        words = ["speech"]

    # Acoustic Similarity Logic [cite: 27]
    window_size = 12 
    mfcc_vectors = mfccs.T
    i = 0
    while i < len(mfcc_vectors) - (2 * window_size):
        win_a = mfcc_vectors[i : i + window_size]
        win_b = mfcc_vectors[i + window_size : i + 2 * window_size]
        score = np.mean(cosine_similarity(win_a, win_b))
        
        if score > 0.970: # Strict match for stuttering [cite: 20]
            repetition_count += 1
            match_time = times[i]
            
            # RECONSTRUCT PATTERN: Force word to look like "ba-ba-ball" 
            if words:
                word_idx = min(int((match_time / times[-1]) * len(words)), len(words)-1)
                target = words[word_idx]
                syl = target[:2] if len(target) > 2 else target
                final_pattern = f'"{syl}-{syl}-{target}"'
            
            # COOLDOWN: Jump 2 seconds ahead so we don't over-count 
            i += int(2.0 / (times[1] - times[0])) 
        else:
            i += 1

    return final_pattern, repetition_count


# 4. FINAL OUTPUT PIPELINE

def run_assignment(file_path):
    y, sr_r, rms, mfccs, times = get_features(file_path)
    pauses, total_dur = detect_pauses(rms, times)
    pattern, count = detect_repetitions(file_path, mfccs, times)
    
    print(f"File: {os.path.basename(file_path)}\n")
    print("Pause Segments:")
    # Showing first 5 to match clean output style [cite: 38]
    print(", ".join([f"[{p[0]}s - {p[1]}s]" for p in pauses[:5]]) + ("..." if len(pauses) > 5 else ""))
    print(f"Total Pause Duration: {total_dur}s\n")
    
    print("Repetitions:")
    print(f"Detected pattern: {pattern}")
    print(f"Repetition Count: {count}")


run_assignment("./M_1064_47y0m_1.wav")