#!/usr/bin/env python3
# scripts/1_convert_logmel.py
# %%
import os
import librosa
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─── 설정 영역 ──────────────────────────────────────────
INPUT_DIR        ="/Users/minimac/Desktop/work/fire_classification/data"           # 원본 .wav 폴더 (positive/, negative/)
OUTPUT_DIR       = "logmel_data"    # 변환된 .npy 저장 폴더
SR               = 22050           # Sampling rate (Hz)
N_MELS           = 64              # Mel band 수
N_FFT            = 2048            # FFT 윈도우 크기
HOP_LENGTH       = 512             # hop length (샘플 단위)
SAVE_AS_FLOAT16  = False           # True면 float16으로 저장 (용량/속도 최적화)
VISUALIZE_SAMPLE = False           # True면 변환 후 샘플 1개 시각화

# ─── 디렉토리 생성 ─────────────────────────────────────
for cls in ["positive", "negative"]:
    os.makedirs(os.path.join(OUTPUT_DIR, cls), exist_ok=True)

# ─── Log-Mel 변환 함수 ──────────────────────────────────
def convert_to_logmel(wav_path):
    """
    .wav 파일 경로를 받아서`
    (1, N_MELS, T) 형태의 Log-Mel Spectrogram numpy array 반환
    """
    y, _ = librosa.load(wav_path, sr=SR)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    logmel = librosa.power_to_db(mel)  # dB 스케일로 변환
    return logmel[np.newaxis, :, :]   # shape: (1, N_MELS, T)

# ─── 메인 실행 ─────────────────────────────────────────
if __name__ == "__main__":
    print("🔁 Log-Mel Spectrogram 변환 시작...")
    for label in ["positive", "negative"]:
        in_dir  = os.path.join(INPUT_DIR,  label)
        out_dir = os.path.join(OUTPUT_DIR, label)

        for fname in tqdm(os.listdir(in_dir), desc=label):
            if not fname.endswith(".wav"):
                continue

            wav_path  = os.path.join(in_dir,  fname)
            save_path = os.path.join(out_dir, fname.replace(".wav", ".npy"))

            # 변환
            logmel = convert_to_logmel(wav_path)

            # 형 변환 (선택 사항)
            if SAVE_AS_FLOAT16:
                logmel = logmel.astype(np.float16)

            # 저장
            np.save(save_path, logmel)

    print("✅ 모든 .wav → Log-Mel 변환 완료!")
#%%

import os
import random
import numpy as np
import matplotlib.pyplot as plt

# ─── 설정 ─────────────────────
logmel_root = "/Users/minimac/Desktop/work/fire_classification/logmel_data"
classes = ["positive", "negative"]
num_samples = 2  # 클래스당 시각화할 샘플 수

# ─── 시각화 ───────────────────
fig, axes = plt.subplots(2, num_samples, figsize=(12, 6))

for row, label in enumerate(classes):
    folder = os.path.join(logmel_root, label)
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    
    if len(files) < num_samples:
        print(f"⚠️ {label} 클래스에 .npy 파일이 부족합니다.")
        continue
    
    samples = random.sample(files, num_samples)
    
    for col, fname in enumerate(samples):
        path = os.path.join(folder, fname)
        data = np.load(path)  # shape: (1, 64, T)

        ax = axes[row, col]
        ax.imshow(data[0], aspect="auto", origin="lower")
        ax.set_title(f"{label} - {fname}", fontsize=10)
        ax.axis("off")

plt.suptitle("Log-Mel Spectrogram Samples", fontsize=14)
plt.tight_layout()
plt.show()



# %%
