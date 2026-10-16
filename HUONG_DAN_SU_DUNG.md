# HƯỚNG DẪN CÀI ĐẶT VÀ SỬ DỤNG DỰ ÁN D2DEWARP & D2DEWARP+

Tài liệu này hướng dẫn chi tiết từng bước để thiết lập môi trường, tải trọng số mô hình, chạy suy luận nắn phẳng ảnh tài liệu (inference) và chạy script đánh giá định lượng (evaluation benchmark).

---

## MỤC LỤC
1. [Yêu cầu môi trường & Cài đặt](#1-yêu-cầu-môi-trường--cài-đặt)
2. [Tải trọng số mô hình (Pretrained Model)](#2-tải-trọng-số-mô-hình-pretrained-model)
3. [Chuẩn bị dữ liệu kiểm thử (Images & Dataset)](#3-chuẩn-bị-dữ-liệu-kiểm-thử-images--dataset)
4. [Hướng dẫn chạy suy luận nắn phẳng (predict.py)](#4-hướng-dẫn-chạy-suy-luận-nắn-phẳng-predictpy)
5. [Hướng dẫn chạy script đánh giá đối sánh (run_benchmark_comparison.py)](#5-hướng-dẫn-chạy-script-đánh-giá-đối-sánh-run_benchmark_comparisonpy)
6. [Cấu trúc thư mục đầu ra (Output Directory Structure)](#6-cấu-trúc-thư-mục-đầu-ra-output-directory-structure)
7. [Xử lý lỗi thường gặp (Troubleshooting)](#7-xử-lý-lỗi-thường-gặp-troubleshooting)

---

## 1. Yêu cầu môi trường & Cài đặt

### Yêu cầu hệ thống:
- **Hệ điều hành:** Windows 10/11, Ubuntu/Linux, hoặc macOS.
- **Python:** Phiên bản `>= 3.8` (đã kiểm thử ổn định trên Python 3.10, 3.11, 3.12, 3.14).
- **Phần cứng:** 
  - Khuyến nghị GPU NVIDIA (tối thiểu 4GB VRAM) có cài đặt CUDA để suy luận nhanh (~0.08s/ảnh).
  - Code hỗ trợ tự động nhận diện và chạy mượt mà trên **CPU** nếu máy không có GPU rời.

### Các bước cài đặt:

1. **Clone repository về máy:**
   ```bash
   git clone https://github.com/nck345/TGMT.git
   cd TGMT
   ```

2. **Khởi tạo và kích hoạt môi trường ảo (khuyến nghị):**
   - Trên Windows:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - Trên Linux/macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```
   *Lưu ý:* Nếu muốn cài đặt PyTorch với phiên bản CUDA phù hợp với GPU của máy bạn, hãy truy cập [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) để lấy lệnh cài đặt tương ứng trước khi chạy lệnh trên.

---

## 2. Tải trọng số mô hình (Pretrained Model)

Mô hình yêu cầu checkpoint trọng số `model_weight.pt` đã được huấn luyện sẵn của D2Dewarp:

- **Link tải chính thức từ tác giả bài báo:**
  - [Google Drive Link](https://drive.google.com/drive/folders/1gv0lLaZWlUt1tX7EJDaVhtFhmV-ewQox?usp=sharing)
  - [Baidu Cloud Link](https://pan.baidu.com/s/1qnHjy3-ANlrrgjc7hac3Bg?pwd=2cut) (Mã truy cập: `2cut`)

- **Vị trí lưu file:**
  Tải file `model_weight.pt` về và đặt trực tiếp tại thư mục gốc của project:
  ```text
  TGMT/
  ├── model_weight.pt        <-- Đặt tại đây
  ├── predict.py
  ├── run_benchmark_comparison.py
  ...
  ```

---

## 3. Chuẩn bị dữ liệu kiểm thử (Images & Dataset)

Bạn có thể chạy kiểm thử trên:
1. **Ảnh tự chọn:** Tạo một thư mục chứa ảnh chụp tài liệu bị cong, nghiêng, nếp gấp, ví dụ: `./demo/` hoặc `./data/test/`.
2. **Tập dữ liệu Benchmark DocUNet (130 ảnh):**
   - Có thể lấy từ thư mục `DocTr-Plus/data benchmark/crop/crop/` hoặc tải bộ benchmark DocUNet chuẩn.
   - Code đã được thiết kế sử dụng đường dẫn tương đối động, tự động định vị tập benchmark nếu đặt repository `DocTr-Plus` cùng cấp thư mục với `TGMT` (hoặc `D2Dewarp`).

---

## 4. Hướng dẫn chạy suy luận nắn phẳng (predict.py)

File `predict.py` dùng để nắn phẳng hình học và khử biến dạng cho một hoặc nhiều ảnh tài liệu.

### Cú pháp cơ bản:

#### Chạy với cải tiến D2Dewarp+ (Mặc định - Khuyến nghị):
Lệnh này sẽ tự động chạy mô hình D2Dewarp và áp dụng các module hậu xử lý của **D2Dewarp+** (cân bằng trắng thích nghi và tự động cắt viền đen):
```bash
python predict.py --model_path ./model_weight.pt --img_path ./demo --save_path ./infer/output --plus True
```

#### Chạy bản D2Dewarp gốc (không áp dụng hậu xử lý cải tiến):
Nếu bạn chỉ muốn xuất kết quả thuần túy của D2Dewarp gốc:
```bash
python predict.py --model_path ./model_weight.pt --img_path ./demo --save_path ./infer/output --plus False
```

### Danh sách các tham số (Arguments):
| Tham số | Giá trị mặc định | Giải thích chi tiết |
| :--- | :---: | :--- |
| `--model_path` | `./model_weight.pt` | Đường dẫn đến file trọng số mô hình. |
| `--img_path` | `./demo` (hoặc auto-detect) | Đường dẫn đến thư mục chứa ảnh cần làm phẳng. |
| `--save_path` | `./infer/DocUNet` | Thư mục lưu kết quả xuất ra. |
| `--plus` | `True` | Bật/tắt cải tiến D2Dewarp+ (Cân bằng trắng thích nghi + Cắt viền). |
| `--input_size` | `448` | Kích thước ảnh đưa vào mạng nơ-ron ($448 \times 448$). |
| `--d_model` | `448` | Chiều đặc trưng của mạng UNet và Attention layer. |
| `--hv_out_chans`| `1` | Số kênh đầu ra của dự đoán dòng ngang (H) và dòng dọc (V). |
| `--in_chans` | `4` | Số kênh đầu vào (3 kênh RGB + 1 kênh Gradient Edge). |

---

## 5. Hướng dẫn chạy script đánh giá đối sánh (run_benchmark_comparison.py)

File `run_benchmark_comparison.py` là công cụ tự động hóa toàn bộ quy trình đánh giá định lượng các chỉ số học thuật:
- **MS-SSIM ($\uparrow$):** Độ tương đồng cấu trúc đa thang đo (Multi-Scale Structural Similarity).
- **CER ($\downarrow$):** Tỷ lệ lỗi nhận dạng ký tự OCR (Character Error Rate).
- **ED ($\downarrow$):** Khoảng cách chỉnh sửa Levenshtein (Edit Distance).

Script tự động đo đạc đối chiếu 4 phương pháp trên các case thực nghiệm:
1. **Distorted:** Ảnh chụp cong biến dạng ban đầu.
2. **Baseline 0:** Phương pháp truyền thống `page-dewarp` (Mô hình mặt trụ bậc ba).
3. **Baseline 1:** Mô hình Deep Learning `DocTr` (ACM MM 2021 / IEEE TMM 2023).
4. **D2Dewarp:** Mô hình đề xuất của chúng ta (CVPR 2026).

### Cách chạy:

#### Chạy mặc định (Tự động nhận diện đường dẫn):
```bash
python run_benchmark_comparison.py
```

#### Chạy với đường dẫn tùy biến:
Nếu bộ dữ liệu DocUNet hoặc DocTr-Plus của bạn được lưu ở một thư mục khác, bạn có thể truyền trực tiếp qua tham số:
```bash
python run_benchmark_comparison.py \
    --model_path ./model_weight.pt \
    --crop_dir "/duong/dan/data_benchmark/crop/crop" \
    --scan_dir "/duong/dan/data_benchmark/scan/scan" \
    --doctr_dir "/duong/dan/DocTr-Plus/results" \
    --save_path ./benchmark_results
```

### Kết quả xuất ra từ script:
1. **Bảng tổng hợp in trực tiếp ra Terminal:**
   ```text
   Sample   | Category       | Method                 | MS-SSIM  | CER      | ED      | Chars        | Time (s)
   ---------------------------------------------------------------------------------------------------------------
   51_1     | Dễ - Vừa       | Distorted              | 0.4512   | 0.5210   | 512     | 470/982      | -       
            |                | Baseline 0 (page-dewarp)| 0.4789  | 0.2840   | 279     | 703/982      | ~12.5   
            |                | Baseline 1 (DocTr)     | 0.5105   | 0.1746   | 171     | 811/982      | ~2.1    
            |                | D2Dewarp (CVPR 2026)   | 0.5284   | 0.1185   | 116     | 866/982      | 0.082   
   ...
   ```
2. **File JSON báo cáo chi tiết:** Lưu tại `benchmark_results/benchmark_comparison_results.json`.
3. **Ảnh so sánh đối chiếu trực quan (5-way panel):** Lưu trong `benchmark_results/visualizations/` gồm 5 cột ảnh: *Distorted -> page-dewarp -> DocTr -> D2Dewarp -> Scan Ground Truth*.

---

## 6. Cấu trúc thư mục đầu ra (Output Directory Structure)

Khi chạy `predict.py`, thư mục đầu ra sẽ có cấu trúc như sau:
```text
infer/output/
├── dewarp/         # Kết quả làm phẳng hình học của D2Dewarp gốc
│   ├── image_1.png
│   └── image_2.png
├── dewarp_plus/    # Kết quả sau cải tiến D2Dewarp+ (khử bóng & cắt viền)
│   ├── image_1.png
│   └── image_2.png
├── pred_h/         # Bản đồ dự đoán đường ngang (Horizontal Lines)
│   └── ...
└── pred_v/         # Bản đồ dự đoán đường dọc (Vertical Lines)
    └── ...
```

---

## 7. Xử lý lỗi thường gặp (Troubleshooting)

1. **Lỗi `FileNotFoundError: model_weight.pt`:**
   - *Nguyên nhân:* Chưa tải file checkpoint trọng số về máy hoặc file để sai thư mục.
   - *Khắc phục:* Tải file theo link ở Mục 2 và đặt vào thư mục gốc dự án.

2. **Lỗi `CUDA out of memory` trên GPU yếu:**
   - *Khắc phục:* Mô hình chỉ chiếm khoảng 1.5GB VRAM khi suy luận. Nếu GPU bị tràn bộ nhớ do tác vụ khác, bạn có thể ép chạy trên CPU bằng cách tắt biến môi trường CUDA:
     ```bash
     set CUDA_VISIBLE_DEVICES=
     python predict.py ...
     ```

3. **Lỗi `weights_only=True` khi load PyTorch:**
   - *Khắc phục:* Code trong repository đã được cấu hình tự động `weights_only=False` tương thích với mọi phiên bản PyTorch mới nhất (>= 2.4+). Bạn không cần chỉnh sửa gì thêm.
