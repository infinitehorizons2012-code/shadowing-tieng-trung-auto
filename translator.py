import os
import json
import jieba
from pypinyin import pinyin, Style
from openai import OpenAI

def generate_pinyin(text):
    result = pinyin(text, style=Style.TONE)
    return [item[0] for item in result]

def translate_text(text_list, api_key):
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )
    
    combined_text = "\n".join(text_list)
    prompt = f"Dịch file phụ đề sau sang tiếng Việt. Đây là video tiếng Trung. Hãy dịch tự nhiên, văn phong nói, ưu tiên dịch thoát ý.\n\n{combined_text}"
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a professional Chinese-Vietnamese translator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        translated_text = response.choices[0].message.content
        return translated_text.split('\n')
    except Exception as e:
        print(f"Error translating: {e}")
        return ["Lỗi dịch thuật"] * len(text_list)

if __name__ == "__main__":
    if not os.path.exists("gpu_analysis.json"):
        print("gpu_analysis.json not found. Did the download from GDrive fail?")
        exit(1)
        
    with open("gpu_analysis.json", "r", encoding="utf-8") as f:
        segments = json.load(f)
        
    sentences = [seg.get("chinese", "") for seg in segments]
    
    # 1. Pinyin Generation
    print("Generating Pinyin...")
    pinyin_list = [generate_pinyin(s) for s in sentences]
    
    # 2. Jieba Word Segmentation (Phân tích cấu tạo từ)
    print("Running Jieba...")
    jieba_list = [list(jieba.cut(s)) for s in sentences]
    
    # 3. Translation
    print("Translating with DeepSeek...")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    translations = []
    if api_key and sentences:
        translations = translate_text(sentences, api_key)
    else:
        print("DEEPSEEK_API_KEY not found. Skipping translation.")
        translations = [""] * len(sentences)
        
    # 4. Merge Data
    final_segments = []
    for idx, seg in enumerate(segments):
        new_seg = {
            "id": seg.get("id"),
            "text": seg.get("chinese", ""),
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "pinyin": pinyin_list[idx],
            "vietnamese": translations[idx] if idx < len(translations) else "",
            "jieba_segmentation": jieba_list[idx]
        }
        final_segments.append(new_seg)
        
    # Save output for the compiler
    with open("linguistic_analysis.json", "w", encoding="utf-8") as f:
        json.dump(final_segments, f, ensure_ascii=False, indent=2)
        
    print("Saved linguistic_analysis.json")
