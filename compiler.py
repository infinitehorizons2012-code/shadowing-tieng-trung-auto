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

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Successfully compiled to {output_path}")

if __name__ == "__main__":
    compile_html()
