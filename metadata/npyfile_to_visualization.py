import os
import numpy as np
import matplotlib.pyplot as plt

# 🔧 설정: .npy 파일이 들어 있는 최상위 디렉토리
root_dir = "/Users/minimac/Desktop/work/fire_classification/logmel_data/positive"   # <- 여기를 수정하세요
output_dir = "/Users/minimac/Desktop/work/fire_classification/metadata/logmel_data_visualization/positive"
os.makedirs(output_dir, exist_ok=True)

# 🔁 모든 .npy 파일을 반복
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith('.npy'):
            npy_path = os.path.join(dirpath, filename)
            try:
                data = np.load(npy_path)

                # ✅ (1, 64, 431) -> (64, 431) 로 변환
                if data.ndim == 3 and data.shape[0] == 1:
                    data = data[0]

                # 시각화
                plt.figure(figsize=(10, 4))
                plt.imshow(data, aspect='auto', origin='lower', cmap='magma')
                title = os.path.relpath(npy_path, root_dir)
                plt.title(title, fontsize=8)
                plt.colorbar()
                plt.tight_layout()

                # 저장 경로 설정
                save_name = title.replace(os.sep, '__').replace('.npy', '.png')
                save_path = os.path.join(output_dir, save_name)

                plt.savefig(save_path)
                plt.close()
                print(f"✅ Saved: {save_path}")
            except Exception as e:
                print(f"❌ Failed: {npy_path} - {e}")