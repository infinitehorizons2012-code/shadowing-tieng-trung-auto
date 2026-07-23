# Hướng dẫn chạy Phần 1 trên Google Colab (Mô hình GPU nặng)

Để đảm bảo hiệu năng và tốc độ cho FunASR và HanLP, bạn hãy mở [Google Colab](https://colab.research.google.com/) và tạo một sổ tay (Notebook) mới.

Hãy chắc chắn bạn vào menu **Runtime > Change runtime type > Chọn T4 GPU** trước khi chạy.

⚠️ **LƯU Ý CỰC KỲ QUAN TRỌNG ĐỂ SỬA LỖI BERTTOKENIZER:**
Sau khi dán và chạy đoạn code ở dưới xong, nếu gặp lỗi `AttributeError: BertTokenizer has no attribute encode_plus`, bạn **BẮT BUỘC** phải vào Menu **Runtime (Thời gian chạy)** -> Chọn **Restart session (Khởi động lại phiên)**. Sau đó bấm chạy lại đoạn code một lần nữa thì mới thành công (do Colab cần xóa thư viện lỗi khỏi bộ nhớ).

Tạo 1 cell và dán toàn bộ đoạn code sau vào chạy:

```python
# 1. Cài đặt các thư viện cần thiết
!pip install funasr modelscope torch torchaudio hanlp yt-dlp gdown

import os
import json
import gdown
from funasr import AutoModel
import transformers

# Vá lỗi (Monkey patch) cho transformers bản mới (Bị thiếu hàm encode_plus mà HanLP cần)
if not hasattr(transformers.PreTrainedTokenizerBase, 'encode_plus'):
    transformers.PreTrainedTokenizerBase.encode_plus = transformers.PreTrainedTokenizerBase.__call__
if getattr(transformers, 'BertTokenizer', None) and not hasattr(transformers.BertTokenizer, 'encode_plus'):
    transformers.BertTokenizer.encode_plus = transformers.BertTokenizer.__call__

import hanlp
from google.colab import drive

# 2. Kết nối Google Drive
drive.mount('/content/drive')

# 3. Tải video từ Link và lưu vĩnh viễn vào Google Drive
# Dán link video của bạn vào đây (Hỗ trợ Youtube, Bilibili, TikTok, Facebook, hoặc link Google Drive)
VIDEO_URL = "DÁN_LINK_VIDEO_CỦA_BẠN_VÀO_ĐÂY"
VIDEO_FILENAME = "/content/drive/MyDrive/video_shadowing.mp4"

print("Đang tải video về Google Drive...")
if "drive.google.com" in VIDEO_URL:
    # Nếu là link Google Drive, dùng gdown
    gdown.download(VIDEO_URL, VIDEO_FILENAME, fuzzy=True)
else:
    # Nếu là link Youtube/Bilibili/Tiktok..., dùng yt-dlp
    os.system(f'yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4" -o "{VIDEO_FILENAME}" "{VIDEO_URL}"')

if not os.path.exists(VIDEO_FILENAME):
    print("LỖI: Tải video thất bại. Xin hãy kiểm tra lại link!")
else:
    print(f"Đã tải xong video và lưu vào: {VIDEO_FILENAME}")
    
    # 4. Chạy FunASR (Bóc băng & mốc thời gian)
    print("Running FunASR...")
    asr_model = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc")
    asr_res = asr_model.generate(input=VIDEO_FILENAME)

    # 5. Chạy HanLP (Phân tích ngữ pháp)
    print("Loading HanLP...")
    HanLP = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)

    print("Analyzing syntax...")
    gpu_analysis_data = []

    # Trích xuất dữ liệu từ kết quả FunASR
    if isinstance(asr_res, list) and len(asr_res) > 0:
        for idx, item in enumerate(asr_res):
            text = item.get("text", "")
            if text:
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

    # 6. Lưu file vào Google Drive
    OUTPUT_PATH = "/content/drive/MyDrive/gpu_analysis.json"
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(gpu_analysis_data, f, ensure_ascii=False, indent=2)

    print(f"XONG! File đã được lưu tại: {OUTPUT_PATH}")
    print("Hãy vào Drive, lấy file gpu_analysis.json chia sẻ dạng 'Bất kỳ ai có liên kết' và dán vào ô input của GitHub Actions nhé!")
```
