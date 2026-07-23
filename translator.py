import os
import json
import jieba
import asyncio
from pypinyin import pinyin, Style
from openai import AsyncOpenAI

def generate_pinyin_str(text):
    result = pinyin(text, style=Style.TONE)
    # Join into a single string for the React app
    return " ".join([item[0] for item in result])

async def process_segment(client, seg):
    text = seg.get("chinese", "")
    if not text:
        return {}

    prompt = f"""Phân tích chi tiết câu tiếng Trung sau: "{text}"
    
Hãy trả về DUY NHẤT một chuỗi JSON hợp lệ với cấu trúc sau, không kèm giải thích:
{{
    "vietnamese": "Dịch nghĩa tiếng Việt tự nhiên, phù hợp văn cảnh",
    "han_nom": "Phân tích Hán Nôm tổng quan cho câu (nếu có, hoặc giải thích sự thú vị về Hán Nôm)",
    "grammar": "Phân tích cấu trúc ngữ pháp chính của câu",
    "words": [
        {{
            "sttTrongCau": 1,
            "char": "Từ 1",
            "am_doc": "Âm Hán Việt",
            "pinyin": "Pinyin của từ",
            "nghia": "Nghĩa tiếng Việt",
            "chiet_tu": "Chiết tự (nếu có)",
            "goc": "",
            "variant": ""
        }}
    ]
}}"""
    try:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "You are a professional Chinese linguistic analyst. Always output valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                content = response.choices[0].message.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                data = json.loads(content)
                return data
            except Exception as e:
                if attempt == 2:
                    raise e
                await asyncio.sleep(2)
    except Exception as e:
        print(f"Error processing segment {seg.get('id')}: {e}")
        return {
            "vietnamese": f"Lỗi: {type(e).__name__} - {str(e)[:50]}",
            "han_nom": "Lỗi phân tích",
            "grammar": "Lỗi phân tích",
            "words": []
        }

async def process_all(segments, api_key):
    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    sem = asyncio.Semaphore(3) # Reduce to 3 concurrent requests to avoid rate limits
    
    async def bounded_process(seg):
        async with sem:
            print(f"Processing id {seg.get('id')}...")
            result = await process_segment(client, seg)
            
            # Format words for React App
            formatted_words = []
            for w in result.get("words", []):
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

            new_seg = {
                "id": seg.get("id"),
                "text": seg.get("chinese", ""),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "pinyin": generate_pinyin_str(seg.get("chinese", "")),
                "vietnamese": result.get("vietnamese", ""),
                "han_nom": result.get("han_nom", ""),
                "grammar": result.get("grammar", ""),
                "words": formatted_words,
                "jieba_segmentation": list(jieba.cut(seg.get("chinese", "")))
            }
            return new_seg

    tasks = [bounded_process(seg) for seg in segments]
    return await asyncio.gather(*tasks)

def main():
    if not os.path.exists("gpu_analysis.json"):
        print("gpu_analysis.json not found.")
        exit(1)
        
    with open("gpu_analysis.json", "r", encoding="utf-8") as f:
        segments = json.load(f)
        
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("DEEPSEEK_API_KEY not found. Skipping translation.")
        # Minimal processing if no key
        final_segments = []
        for seg in segments:
            final_segments.append({
                "id": seg.get("id"),
                "text": seg.get("chinese", ""),
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "pinyin": generate_pinyin_str(seg.get("chinese", "")),
                "vietnamese": "",
                "han_nom": "",
                "grammar": "",
                "words": [],
                "jieba_segmentation": list(jieba.cut(seg.get("chinese", "")))
            })
    else:
        # Run async processing
        final_segments = asyncio.run(process_all(segments, api_key))
        
    with open("linguistic_analysis.json", "w", encoding="utf-8") as f:
        json.dump(final_segments, f, ensure_ascii=False, indent=2)
        
    print("Saved linguistic_analysis.json")

if __name__ == "__main__":
    main()
