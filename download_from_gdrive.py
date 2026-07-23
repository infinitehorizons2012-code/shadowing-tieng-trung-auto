import os
import re
import gdown

def extract_file_id(url):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    match = re.search(r'id=([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return url

if __name__ == "__main__":
    gdrive_link = os.environ.get("GDRIVE_FILE_LINK")
    
    if not gdrive_link:
        print("Missing GDRIVE_FILE_LINK environment variable.")
        exit(1)
        
    file_id = extract_file_id(gdrive_link)
    print(f"Attempting to download public GDrive file ID: {file_id}")
    
    output = "gpu_analysis.json"
    # Using gdown to download public google drive file
    download_url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(download_url, output, quiet=False)
    
    if os.path.exists(output):
        print(f"Successfully downloaded {output}")
    else:
        print("Failed to download. Please ensure the file is shared as 'Anyone with the link'.")
