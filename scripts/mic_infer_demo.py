#!/usr/bin/env python3
# mic_infer_demo.py — 실시간 마이크 → log-mel → MobileNetV2 분류 → 배너 표시 (2초마다 한 번만 추론)

import os, queue, time
from collections import deque
import numpy as np
import sounddevice as sd
import librosa
import torch
import torch.nn.functional as F
import timm
import tkinter as tk

# ====== 경로 설정 ======
BASE_DIR = "/Users/minimac/Desktop/work/fire_classification"
MODEL_PATH = os.path.join(BASE_DIR, "models", "mobilenetv2_050_exp001.pt")
CLASS = ["일상음", "경보음"]

# ====== 오디오/스펙트로그램 설정 ======
SR = 22050
N_MELS = 64
N_FFT = 2048
HOP_LENGTH = 512
WIN_SEC = 1.0
HOP_SEC = 0.25
THRESH = 0.80
DEBOUNCE = 1.5
INFER_INTERVAL = 2.0

# ====== 배너 UI 설정 ======
root = tk.Tk()
root.title("경보음 감지 데모")
root.geometry("720x220")
root.configure(bg="#111111")
label = tk.Label(root, text="대기 중…", fg="#DDDDDD", bg="#111111",
                 font=("Apple SD Gothic Neo", 42, "bold"))
label.pack(expand=True, fill="both")

def set_banner(pred: int, prob: float):
    if pred == 1 and prob >= THRESH:
        label.config(text=f"🚨 경보음 감지!  p={prob:.2f}", bg="#C10015", fg="#FFFFFF")
    else:
        label.config(text=f"정상  p={prob:.2f}", bg="#1E1E1E", fg="#A0FFA0")
    root.update_idletasks()

# ====== 모델 로드 ======
def load_mobilenet():
    model = timm.create_model(
        "mobilenetv2_050", pretrained=False,
        in_chans=1, num_classes=2
    )
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

# ====== log-mel 변환 ======
def wav_to_logmelspec(y, sr=SR):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )
    logmel = librosa.power_to_db(mel)
    return logmel[np.newaxis, :, :]  # shape: (1, 64, T)

# ====== 메인 루프 ======
def main():
    print(f"[INFO] MODEL_PATH: {MODEL_PATH}")
    model = load_mobilenet()
    print("[INFO] 모델 로드 완료.")

    q = queue.Queue()
    block = int(SR * HOP_SEC)
    win = int(SR * WIN_SEC)
    ring = deque(maxlen=win)
    last_ts = 0.0
    last_state = 0  # 클래스 index 저장
    last_infer_time = 0.0

    def audio_cb(indata, frames, time_info, status):
        if status: print("[WARN]", status)
        q.put(indata.copy())

    print("[INFO] 실시간 마이크 수집 시작 (종료: Ctrl+C)")
    set_banner(0, 0.0)

    with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                        blocksize=block, callback=audio_cb):
        while True:
            buf = q.get()
            mono = buf[:, 0] if buf.ndim > 1 else buf
            ring.extend(mono)

            if len(ring) < win:
                root.update(); continue

            y = np.array(ring)
            if np.sqrt(np.mean(y**2)) < 1e-5:
                set_banner(0, 0.0); root.update(); continue

            # ✅ 2초에 한 번만 추론하도록 제한
            now = time.time()
            if now - last_infer_time < INFER_INTERVAL:
                root.update(); continue
            last_infer_time = now

            # log-mel 추론
            melspec = wav_to_logmelspec(y, sr=SR)
            x = torch.tensor(melspec, dtype=torch.float32).unsqueeze(0)  # (1,1,64,T)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=-1).cpu().numpy().squeeze()

            pred = int(np.argmax(probs))
            prob = float(probs[pred])

            print(f"[{time.strftime('%H:%M:%S')}] {CLASS[pred]} (p={prob:.2f})")

            # ✅ 올바르게 pred 기반으로 UI 업데이트
            if pred != last_state and (now - last_ts) > DEBOUNCE:
                set_banner(pred, prob)
                last_state = pred
                last_ts = now
            else:
                set_banner(last_state, prob)

            root.update()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n종료합니다.")
