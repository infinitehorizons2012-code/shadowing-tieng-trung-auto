# Hướng dẫn chạy Phần 1 trên Google Colab (Mô hình GPU nặng)

Để đảm bảo hiệu năng và tốc độ cho FunASR và HanLP, bạn hãy mở [Google Colab](https://colab.research.google.com/) và tạo một sổ tay (Notebook) mới.

Hãy chắc chắn bạn vào menu **Runtime > Change runtime type > Chọn T4 GPU** trước khi chạy.

Tạo 1 cell và dán toàn bộ đoạn code sau vào chạy:

```python
# 1. Cài đặt các thư viện cần thiết
!pip install funasr modelscope torch torchaudio hanlp

import os
import json
import urllib.request
from funasr import AutoModel
import hanlp
from google.colab import drive

# 2. Kết nối Google Drive để lưu file đầu ra
drive.mount('/content/drive')

# 3. Tải video (thay thế URL bằng link video thực tế của bạn)
VIDEO_URL = "LINK_VIDEO_CUA_BAN_O_DAY"
VIDEO_FILENAME = "video.mp4"
if VIDEO_URL != "LINK_VIDEO_CUA_BAN_O_DAY":
    print("Downloading video...")
    urllib.request.urlretrieve(VIDEO_URL, VIDEO_FILENAME)

# 4. Chạy FunASR (Bóc băng & mốc thời gian)
print("Running FunASR...")
asr_model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc")
asr_res = asr_model.generate(input=VIDEO_FILENAME)

# 5. Chạy HanLP (Phân tích ngữ pháp)
print("Loading HanLP...")
# Tải mô hình đa tác vụ chuẩn của HanLP
HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)

print("Analyzing syntax...")
gpu_analysis_data = []

# Trích xuất dữ liệu từ kết quả FunASR (ASR trả về list các dict)
if isinstance(asr_res, list) and len(asr_res) > 0:
    for idx, item in enumerate(asr_res):
        text = item.get("text", "")
        if text:
            # Phân tích bằng HanLP
            doc = HanLP(text)
            
            segment_data = {
                "id": idx,
                "chinese": text,
                "hanlp_analysis": {
                    "tokens": doc.get('tok/fine', []),
                    "pos_tags": doc.get('pos/ctb', []),
                    "dependency": doc.get('dep', [])
                }
            }
            gpu_analysis_data.append(segment_data)

# 6. Lưu file vào Google Drive của bạn
OUTPUT_PATH = "/content/drive/MyDrive/gpu_analysis.json"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(gpu_analysis_data, f, ensure_ascii=False, indent=2)

print(f"XONG! File đã được lưu tại: {OUTPUT_PATH}")
print("Hãy lấy file gpu_analysis.json này chia sẻ dạng link (Bất kỳ ai có liên kết) và dán vào ô input của GitHub Actions nhé!")
```
