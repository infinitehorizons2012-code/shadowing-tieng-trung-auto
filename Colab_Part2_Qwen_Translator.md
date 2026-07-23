# Hướng dẫn chạy Phần 2 trên Google Colab (Dịch thuật bằng Qwen 2.5)

Đây là bước tiếp nối sau khi bạn đã có file `gpu_analysis.json` từ Bước 1 (FunASR). Ở bước này, chúng ta sẽ cài đặt mô hình trí tuệ nhân tạo nguồn mở **Qwen 2.5 (7 Tỷ tham số)** ngay trên Google Colab để tự động dịch thuật, phân tích ngữ pháp, Hán Nôm và chiết tự từ vựng.

**Bước chuẩn bị:**
1. Mở một [Google Colab](https://colab.research.google.com/) mới.
2. Hãy chắc chắn bạn vào menu **Runtime > Change runtime type > Chọn T4 GPU**.

---

### Chạy mã nguồn

Tạo 1 cell và dán toàn bộ đoạn code sau vào chạy:

```python
# 1. Cài đặt các thư viện cần thiết
!pip install -q transformers accelerate bitsandbytes pypinyin jieba gdown autoawq

import os
import json
import re
import jieba
import torch
from pypinyin import pinyin, Style
from transformers import AutoModelForCausalLM, AutoTokenizer
from google.colab import drive

# 2. Kết nối Google Drive
drive.mount('/content/drive')

# Thay thế bằng ID file gpu_analysis.json của bạn trên Google Drive nếu bạn muốn tự động tải, 
# hoặc chỉ cần đảm bảo file gpu_analysis.json đã nằm sẵn trong Drive của bạn.
INPUT_FILE = "/content/drive/MyDrive/gpu_analysis.json"
OUTPUT_FILE = "/content/drive/MyDrive/linguistic_analysis.json"

if not os.path.exists(INPUT_FILE):
    print(f"LỖI: Không tìm thấy file {INPUT_FILE}.")
    print("Vui lòng chạy lại Phần 1 hoặc kiểm tra lại file trong Drive!")
    import sys
    sys.exit()

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    segments = json.load(f)

print("Đang tải mô hình Qwen 2.5 (phiên bản AWQ tối ưu hóa cho T4)... Có thể mất 1-2 phút.")
model_id = "Qwen/Qwen2.5-7B-Instruct-AWQ"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
)

def generate_pinyin_str(text):
    result = pinyin(text, style=Style.TONE)
    return " ".join([item[0] for item in result])

def parse_json_from_output(content):
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    try:
        return json.loads(content)
    except Exception as e:
        print(f"Lỗi phân tích JSON: {e}")
        return None

final_segments = []

for idx, seg in enumerate(segments):
    text = seg.get("chinese", "").strip()
    if not text:
        continue
        
    print(f"Đang xử lý câu {idx + 1}/{len(segments)}: {text}")
    
    prompt = f"""Phân tích chi tiết câu tiếng Trung sau: "{text}"
    
Hãy trả về DUY NHẤT một chuỗi JSON hợp lệ với cấu trúc sau, tuyệt đối không kèm bất kỳ lời giải thích nào ngoài JSON.
Bắt buộc dùng ngoặc kép (") cho key và string.
{{
    "vietnamese": "Dịch nghĩa tiếng Việt tự nhiên, phù hợp văn cảnh",
    "han_nom": "Phân tích Hán Nôm tổng quan cho câu (giải thích sự thú vị về Hán Nôm)",
    "grammar": "Phân tích cấu trúc ngữ pháp chính của câu",
    "words": [
        {{
            "sttTrongCau": 1,
            "char": "Từ 1",
            "am_doc": "Âm Hán Việt",
            "pinyin": "Pinyin của từ",
            "nghia": "Nghĩa tiếng Việt",
            "chiet_tu": "Chiết tự (nếu có, không có để rỗng)",
            "goc": "",
            "variant": ""
        }}
    ]
}}"""
    
    messages = [
        {"role": "system", "content": "You are a professional Chinese linguistic analyst. Always output strictly valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text_prompt], return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.3,
        do_sample=True
    )
    
    # Lấy phần chữ do AI sinh ra (bỏ qua phần prompt)
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    parsed_data = parse_json_from_output(generated_text)
    
    if not parsed_data:
        parsed_data = {
            "vietnamese": "Lỗi AI xuất định dạng",
            "han_nom": "Lỗi",
            "grammar": "Lỗi",
            "words": []
        }
        
    # Chuẩn hóa từ điển cho giao diện
    formatted_words = []
    for w in parsed_data.get("words", []):
        formatted_words.append({
            "sttTrongCau": w.get("sttTrongCau", 0),
            "char": w.get("char", ""),
            "info": {
                "stt": str(w.get("sttTrongCau", 0)),
                "am_doc": w.get("am_doc", ""),
                "pinyin": w.get("pinyin", ""),
                "nghia": w.get("nghia", ""),
                "chiet_tu": w.get("chiet_tu", ""),
                "goc": w.get("goc", ""),
                "variant": w.get("variant", "")
            }
        })
        
    final_segments.append({
        "id": seg.get("id"),
        "text": text,
        "start": seg.get("start", 0),
        "end": seg.get("end", 0),
        "pinyin": generate_pinyin_str(text),
        "vietnamese": parsed_data.get("vietnamese", ""),
        "han_nom": parsed_data.get("han_nom", ""),
        "grammar": parsed_data.get("grammar", ""),
        "words": formatted_words,
        "jieba_segmentation": list(jieba.cut(text))
    })

# Lưu kết quả
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(final_segments, f, ensure_ascii=False, indent=2)

print(f"\n✅ XONG! File đã được lưu tại: {OUTPUT_FILE}")
print("Hãy lấy link của file này dán vào GitHub Actions để xuất ứng dụng nhé!")
```
