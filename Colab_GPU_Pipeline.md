# Hướng dẫn chạy Phần 1 trên Google Colab (Bóc băng FunASR - Hỗ trợ CPU)

Để bóc băng tiếng Trung bằng FunASR, bạn hãy mở [Google Colab](https://colab.research.google.com/) và tạo một sổ tay (Notebook) mới.

Bạn **không cần** phải bật GPU (có thể dùng CPU mặc định của Colab để tiết kiệm giới hạn sử dụng GPU cho Phần 2).

⚠️ **LƯU Ý CỰC KỲ QUAN TRỌNG ĐỂ SỬA LỖI BERTTOKENIZER:**
Sau khi dán và chạy đoạn code ở dưới xong, nếu gặp lỗi `AttributeError: BertTokenizer has no attribute encode_plus`, bạn **BẮT BUỘC** phải vào Menu **Runtime (Thời gian chạy)** -> Chọn **Restart session (Khởi động lại phiên)**. Sau đó bấm chạy lại đoạn code một lần nữa thì mới thành công (do Colab cần xóa thư viện lỗi khỏi bộ nhớ).

Tạo 1 cell và dán toàn bộ đoạn code sau vào chạy:

```python
# 1. Cài đặt các thư viện cần thiết
!pip install funasr modelscope torch torchaudio yt-dlp gdown

import os
import json
import gdown
from funasr import AutoModel
from google.colab import drive

# 2. Kết nối Google Drive
drive.mount('/content/drive')

# 3. Tải video từ Link và lưu vĩnh viễn vào Google Drive
VIDEO_URL = "DÁN_LINK_VIDEO_CỦA_BẠN_VÀO_ĐÂY"
VIDEO_FILENAME = "/content/drive/MyDrive/video_shadowing.mp4"

print("Đang tải video về Google Drive...")
if "drive.google.com" in VIDEO_URL:
    gdown.download(VIDEO_URL, VIDEO_FILENAME, fuzzy=True)
else:
    os.system(f'yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" -o "{VIDEO_FILENAME}" "{VIDEO_URL}"')

if not os.path.exists(VIDEO_FILENAME):
    print("LỖI: Tải video thất bại. Xin hãy kiểm tra lại link!")
else:
    print(f"Đã tải xong video và lưu vào: {VIDEO_FILENAME}")
    
    # 4. Chạy FunASR (Bóc băng & mốc thời gian trên CPU)
    print("Running FunASR on CPU...")
    asr_model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", device="cpu")
    asr_res = asr_model.generate(input=VIDEO_FILENAME, batch_size_s=300, sentence_timestamp=True)

    print("Đang bóc tách mốc thời gian...")
    gpu_analysis_data = []

    if isinstance(asr_res, list) and len(asr_res) > 0:
        res_dict = asr_res[0]
        if "sentence_info" in res_dict:
            for idx, seg in enumerate(res_dict["sentence_info"]):
                text = seg.get("text", "").strip()
                if text:
                    segment_data = {
                        "id": idx,
                        "chinese": text,
                        "start": seg.get("start", 0) / 1000.0,
                        "end": seg.get("end", 0) / 1000.0
                    }
                    gpu_analysis_data.append(segment_data)
        else:
            # Fallback to character-level timestamps if sentence_info is missing
            import re
            full_text = res_dict.get("text", "")
            char_timestamps = res_dict.get("timestamp", [])
            sentences = [s.strip() for s in re.split(r'(?<=[。！？!?])', full_text) if s.strip()]
            
            char_idx = 0
            for idx, sentence in enumerate(sentences):
                # Calculate how many valid characters are in this sentence (ignoring spaces/punctuation if needed)
                # FunASR timestamps typically align with the text including punctuation.
                s_len = len(sentence)
                
                start_time = 0.0
                end_time = 0.0
                
                if char_idx < len(char_timestamps):
                    start_time = char_timestamps[char_idx][0] / 1000.0
                    
                char_idx += s_len
                
                if char_idx - 1 < len(char_timestamps) and char_idx > 0:
                    end_time = char_timestamps[char_idx - 1][1] / 1000.0
                elif len(char_timestamps) > 0:
                    end_time = char_timestamps[-1][1] / 1000.0
                else:
                    # Absolute fallback
                    start_time = idx * 2.0
                    end_time = idx * 2.0 + 2.0
                    
                gpu_analysis_data.append({
                    "id": idx, 
                    "chinese": sentence,
                    "start": start_time,
                    "end": end_time
                })

    # 5. Lưu file vào Google Drive
    OUTPUT_PATH = "/content/drive/MyDrive/gpu_analysis.json"
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(gpu_analysis_data, f, ensure_ascii=False, indent=2)

    print(f"XONG! File đã được lưu tại: {OUTPUT_PATH}")
    print("Bạn có thể tắt Notebook này và chuyển sang chạy Phần 2 (cần bật GPU T4) nhé!")
```
