# Hướng dẫn Thiết lập Shadowing Pipeline (Kiến trúc Hybrid)

Dự án này sử dụng kiến trúc lai cực kỳ tối ưu:
- **Google Colab (GPU)**: Đảm nhận các tác vụ trí tuệ nhân tạo nặng (Bóc băng FunASR & Phân tích cấu trúc câu HanLP).
- **GitHub Actions (CPU)**: Đảm nhận các tác vụ gọi API dịch thuật (DeepSeek), tạo Pinyin, cắt từ (Jieba) và tự động đóng gói Giao diện Web.

## BƯỚC 1: Chạy Google Colab (Lấy Data thô)
1. Hãy mở thư mục dự án trên máy tính của bạn, mở file `Colab_GPU_Pipeline.md`.
2. Truy cập [Google Colab](https://colab.research.google.com/), tạo Notebook mới.
3. Chép đoạn code trong file đó vào và chạy (Nhớ bật GPU T4).
4. Code sẽ tự động tạo file `gpu_analysis.json` nằm trong Google Drive của bạn!
5. Hãy click chuột phải vào file `gpu_analysis.json` trên Drive, chọn **Share (Chia sẻ)**, đặt quyền là **Bất kỳ ai có liên kết (Anyone with the link)** và copy cái Link đó lại.

## BƯỚC 2: Cấu hình GitHub Actions
1. Đẩy toàn bộ mã nguồn ở thư mục này lên GitHub cá nhân của bạn.
2. Vào trang GitHub Repository > **Settings** > **Secrets and variables** > **Actions**.
3. Bấm **New repository secret** để khai báo API Key dịch thuật:
   - **Name:** `DEEPSEEK_API_KEY`
   - **Secret:** (Dán API Key của DeepSeek vào đây).
*(Lưu ý: Do hiện tại chúng ta dùng link Drive chia sẻ công khai ở Bước 1 nên bạn KHÔNG CẦN phải setup GDRIVE_CREDENTIALS phức tạp nữa. Script sẽ tự lấy bằng link bạn nhập!)*

## BƯỚC 3: Kích hoạt Pipeline
1. Vào tab **Actions** trên GitHub.
2. Chọn workflow **Chinese Shadowing Pipeline (Hybrid)** bên tay trái.
3. Bấm **Run workflow**.
4. Dán **Link Google Drive** của file `gpu_analysis.json` (mà bạn đã copy ở Bước 1) vào ô trống.
5. Bấm Run. Hệ thống sẽ tự động tải file đó, dịch sang tiếng Việt, tạo Pinyin, phân tích từ vựng Jieba và chèn thẳng vào file Giao diện `template.html` của bạn!
6. Khi chạy xong, vào mục **Artifacts** tải `build.zip` về là xong!
