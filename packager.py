import os
import zipfile

if __name__ == "__main__":
    files_to_zip = ["shadowing_course.html", "linguistic_analysis.json", "translated_transcript.json", "raw_transcript.json"]
    
    with zipfile.ZipFile("build.zip", 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            if os.path.exists(file):
                zipf.write(file)
                
    print("Packaged HTML and JSON outputs to build.zip")
