import os
import json

def compile_html():
    json_path = "linguistic_analysis.json"
    template_path = "template.html"
    output_path = "shadowing_course.html"

    if not os.path.exists(json_path):
        print(f"{json_path} not found.")
        return

    if not os.path.exists(template_path):
        print(f"{template_path} not found. Please provide the template.html file.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Inject data into <head>
    data_script = f"<script>window.LESSON_DATA = {json.dumps(data, ensure_ascii=False)};</script>\n"
    
    # We find the </head> tag and insert right before it
    if "</head>" in html_content:
        html_content = html_content.replace("</head>", data_script + "</head>")
    else:
        # Fallback if no head tag is found
        html_content = data_script + html_content
        
    # Monkey-patch the React app's hardcoded mock data to use window.LESSON_DATA
    start_idx = html_content.find('JSON.parse(`[{"id":0')
    if start_idx != -1:
        end_idx = html_content.find(']`)', start_idx)
        if end_idx != -1:
            mock_data_str = html_content[start_idx:end_idx+3]
            # Replace the hardcoded parse with a check for our injected data
            replacement = f"(window.LESSON_DATA && window.LESSON_DATA.analysis ? window.LESSON_DATA.analysis : {mock_data_str})"
            html_content = html_content.replace(mock_data_str, replacement)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Successfully compiled to {output_path}")

if __name__ == "__main__":
    compile_html()
