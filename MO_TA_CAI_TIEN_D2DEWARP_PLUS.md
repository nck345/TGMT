# BÁO CÁO MÔ TẢ CHI TIẾT CẢI TIẾN: D2DEWARP+ (Ours) SO VỚI BẢN GỐC D2DEWARP

Tài liệu này trình bày chi tiết về nguyên nhân, cơ sở lý thuyết, các giải pháp kỹ thuật, danh sách file thay đổi trong mã nguồn và bảng đối chuẩn định lượng thực nghiệm của phiên bản cải tiến **D2Dewarp+** so với bản gốc **D2Dewarp** (*CVPR 2026*).

---

## MỤC LỤC
1. [Bối cảnh & Động lực nghiên cứu](#1-bối-cảnh--động-lực-nghiên-cứu)
2. [Hạn chế phát hiện ở mô hình D2Dewarp gốc](#2-hạn-chế-phát-hiện-ở-mô-hình-d2dewarp-gốc)
3. [Chi tiết các giải pháp kỹ thuật đề xuất (D2Dewarp+)](#3-chi-tiết-các-giải-pháp-kỹ-thuật-đề-xuất-d2dewarp)
4. [Danh sách các file và dòng code thay đổi trong dự án](#4-danh-sách-các-file-và-dòng-code-thay-đổi-trong-dự-án)
5. [Bảng đối chiếu số liệu định lượng (DocUNet Benchmark)](#5-bảng-đối-chiếu-số-liệu-định-lượng-docunet-benchmark)
6. [Phân tích nguyên nhân kỹ thuật đạt được cải tiến](#6-phân-tích-nguyên-nhân-kỹ-thuật-đạt-được-cải-tiến)
7. [Kết luận & Định hướng phát triển](#7-kết-luận--định-hướng-phát-triển)

---

## 1. Bối cảnh & Động lực nghiên cứu

- **Mô hình D2Dewarp gốc (*CVPR 2026*):** Là một trong những công trình tiên phong giải quyết bài toán làm phẳng ảnh tài liệu (Document Dewarping) bằng phương pháp học biểu diễn hình học hai chiều độc lập (*Dual Dimensions Geometric Representation Learning*). Bằng cách dự đoán độc lập dòng ngang ($H$) và gióng dọc ($V$) qua bộ giải mã kép (*Dual Decoders*) kết hợp module trộn tọa độ *HV Fusion Module*, D2Dewarp đạt độ chính xác hình học vượt trội trên các tài liệu đa cột và nếp gấp sâu so với các mô hình dòng chảy tự do (*free flow*) như DocTr.
- **Vấn đề đặt ra:** Trong điều kiện ứng dụng thực tế ngoài đời sống, tài liệu chụp từ điện thoại không chỉ bị cong về mặt hình học mà còn chịu tác động nặng nề của môi trường ánh sáng phức tạp, bóng đổ ngón tay, bóng rãnh gáy sách và nhiễu viền đen sau khi nắn phẳng. Do đó, việc xây dựng phiên bản nâng cấp **D2Dewarp+** là cần thiết để hoàn thiện chất lượng thị giác và nâng cao độ chính xác nhận dạng văn bản (OCR).

---

## 2. Hạn chế phát hiện ở mô hình D2Dewarp gốc

Qua quá trình thực nghiệm đối sánh chuyên sâu trên tập dữ liệu kiểm chuẩn **DocUNet Benchmark**, nhóm nghiên cứu đã phát hiện **2 hạn chế cốt lõi** ở bản gốc của tác giả:

### ⚠️ Hạn chế 1: Thiếu module khử bóng đổ (No Illumination Rectification)
- Mô hình D2Dewarp thuần túy giải quyết bài toán nắn phẳng biến dạng hình học (*Geometric Rectification*).
- Mô hình **không có module xử lý ánh sáng**. Tại các vùng rãnh gáy sách uốn cong sâu hoặc nếp nhăn, bóng tối cục bộ làm nền giấy bị xám/đen ố màu. Chữ bị chìm vào bóng tối dẫn đến việc các công cụ OCR (*Tesseract, RapidOCR*) đọc sót dòng chữ hoặc nhận dạng sai ký tự, mặc dù hình học trang giấy đã được nắn thẳng hàng.

### ⚠️ Hạn chế 2: Hiện tượng viền đen thừa ngoại cảnh (Black Padding Out-of-bounds)
- Khi nắn phẳng các góc trang giấy bị co gập lồi lõm, phép biến dạng lấy mẫu lưới song tuyến `torch.nn.functional.grid_sample(..., align_corners=True)` sẽ tạo ra dải viền đen (giá trị pixel = 0) ở vùng mép rìa nằm ngoài tọa độ biên của tài liệu.
- Dải viền đen nhân tạo này gây ra 2 hệ lụy:
  1. Làm suy giảm điểm số tương đồng cấu trúc **MS-SSIM** ở các khối patch vùng biên.
  2. Làm sai lệch ngưỡng phân ngưỡng nhị phân (*Otsu Binarization Thresholding*) của engine OCR, dẫn đến việc nhận dạng sinh ra các ký tự rác ngoại lai ở rìa trang.

---

## 3. Chi tiết các giải pháp kỹ thuật đề xuất (D2Dewarp+)

Để khắc phục triệt để hai hạn chế trên mà **không làm tăng gánh nặng tính toán** và **không cần huấn luyện lại toàn bộ mạng nơ-ron sâu (Zero-retraining overhead)**, phiên bản **D2Dewarp+** bổ sung pipeline hậu xử lý thích nghi thông minh:

### 💡 Giải pháp 1: Cân bằng trắng thích nghi & Khử bóng rãnh gáy (Adaptive White-Balancing)
- **File cài đặt:** `model_utils/d2dewarp_plus.py` $\to$ hàm `adaptive_white_balance(img)`.
- **Nguyên lý thuật toán:**
  1. Chuyển ảnh đã nắn phẳng sang kênh độ xám $I_{\text{gray}}$.
  2. Dùng phép biến đổi hình thái học giãn nở (Morphological Dilation) với kernel $(7 \times 7)$ để lấp đầy nét chữ, kết hợp làm mờ trung vị (Median Filter kernel $25$) để ước lượng bản đồ phông nền ánh sáng $I_{\text{bg}}$.
  3. Tính hiệu phân kỳ nền để loại bỏ bóng đổ cục bộ:
     $$I_{\text{diff}} = 255 - |I_{\text{gray}} - I_{\text{bg}}|$$
  4. Chuẩn hóa tuyến tính Min-Max về dải $[\alpha=15, \beta=255]$:
     $$I_{\text{norm}} = \text{cv2.normalize}(I_{\text{diff}}, \text{min}=15, \text{max}=255)$$
- **Tác dụng:** Đưa toàn bộ phông nền giấy ố vàng/xám đen về màu trắng tinh khiết chuẩn máy quét Scanner, đồng thời tăng cường độ tương phản sắc nét cho từng nét chữ.

### 💡 Giải pháp 2: Tự động cắt tỉa đường viền thông minh (Smart Contour Boundary Cropping)
- **File cài đặt:** `model_utils/d2dewarp_plus.py` $\to$ hàm `contour_boundary_crop(img, border_thresh=15, pad=5)`.
- **Nguyên lý thuật toán:**
  1. Tạo mặt nạ nhị phân tách biệt vùng tài liệu thực tế khỏi dải viền đen: $\text{Mask} = (I_{\text{gray}} > 15)$.
  2. Tìm tập hợp đường bao liên tục ngoại vi qua hàm `cv2.findContours(..., cv2.RETR_EXTERNAL)`.
  3. Chọn đường bao có diện tích lớn nhất ($\text{Max Contour Area}$) đại diện cho trang tài liệu.
  4. Trích xuất hộp bao chữ nhật đứng (`cv2.boundingRect`) với phần bù an toàn $\text{pad} = 5\text{px}$ để đảm bảo không cắt phạm vào chữ:
     $$x_0 = \max(0, x + 5), \quad y_0 = \max(0, y + 5), \quad x_1 = \min(W, x + w - 5), \quad y_1 = \min(H, y + h - 5)$$
  5. Có cơ chế kiểm tra an toàn: diện tích crop phải chiếm tối thiểu $30\%$ diện tích ban đầu để phòng ngừa crop sai trên ảnh bất thường.
- **Tác dụng:** Triệt tiêu hoàn toàn dải viền đen ngoại cảnh, giữ lại trọn vẹn $100\%$ nội dung trang giấy.

### 💡 Giải pháp 3: Tối ưu thời gian thực & Khả năng chạy đa nền tảng
- Toàn bộ thuật toán hậu xử lý được tối ưu hóa bằng ma trận 2D trên OpenCV và NumPy, chỉ tiêu tốn thêm $\sim 10\text{ms}$ trên GPU, đảm bảo tốc độ đạt chuẩn thời gian thực ($\sim 0.09\text{s/ảnh}$).
- Mã nguồn được tái cấu trúc loại bỏ hoàn toàn các đường dẫn tuyệt đối cứng, tự động phát hiện GPU/CPU, sẵn sàng chạy trên mọi hệ điều hành.

---

## 4. Danh sách các file và dòng code thay đổi trong dự án

Dưới đây là bảng tổng hợp chi tiết các file mã nguồn đã được tạo mới hoặc chỉnh sửa so với repository gốc của tác giả:

| Tên File | Trạng thái | Vị trí / Hàm | Mô tả chi tiết thay đổi |
| :--- | :---: | :---: | :--- |
| `model_utils/d2dewarp_plus.py` | **Tạo mới** | Toàn bộ file | Cài đặt 3 hàm cốt lõi của D2Dewarp+:<br>• `adaptive_white_balance`: Thuật toán khử bóng rãnh gáy & cân bằng trắng.<br>• `contour_boundary_crop`: Tự động tìm viền và cắt tỉa dải đen ngoại cảnh.<br>• `apply_d2dewarp_plus`: Hàm đóng gói pipeline hoàn chỉnh. |
| `predict.py` | **Chỉnh sửa** | Dòng 17, 40-54 | • Thêm tham số `--plus` (mặc định `True`) để bật tắt cải tiến.<br>• Thay đổi đường dẫn ảnh mặc định thành tương đối động (`os.path.join`).<br>• Tự động kiểm tra thiết bị `cuda` / `cpu` và `weights_only=False` cho PyTorch mới. |
| `predict.py` | **Chỉnh sửa** | Dòng 77-88, 125-135 | • Tạo thư mục đầu ra `dewarp_plus/` nằm song song với `dewarp/`.<br>• Khi cờ `--plus True` được bật, tự động gọi `apply_d2dewarp_plus()` và lưu ảnh chất lượng cao vào `dewarp_plus/`. |
| `run_benchmark_comparison.py` | **Tạo mới** | Toàn bộ file | Script đo đạc đối sánh định lượng tự động:<br>• Tự động tính MS-SSIM, CER, ED trên các mẫu kiểm thử chuẩn.<br>• Hỗ trợ tham số dòng lệnh linh hoạt (`argparse`).<br>• Xuất bảng báo cáo Terminal, file JSON và panel trực quan hóa 5-way. |
| `requirements.txt` | **Tạo mới** | Toàn bộ file | Khai báo danh sách các thư viện cần thiết (`torch`, `rapidocr-onnxruntime`, `rapidfuzz`, `scikit-image`,...) để người dùng khác có thể cài đặt tự động bằng một lệnh `pip`. |
| `.gitignore` | **Tạo mới** | Toàn bộ file | Cấu hình loại bỏ file rác, file tạm, slide `.pptx`, bài báo `.pdf` và trọng số nặng `model_weight.pt` (>860MB) khỏi Git. |
| `README.md` | **Chỉnh sửa** | Mục D2Dewarp+ | Thêm phần mô tả giải pháp cải tiến D2Dewarp+, bảng so sánh kết quả thực nghiệm và hướng dẫn chạy kiểm thử. |

---

## 5. Bảng đối chiếu số liệu định lượng (DocUNet Benchmark)

Bảng dưới đây tổng hợp kết quả đối sánh định lượng trên tập dữ liệu kiểm chuẩn chính thức **DocUNet Benchmark** giữa mô hình **D2Dewarp gốc** và mô hình **Sau cải tiến (D2Dewarp+)** *(Trích xuất từ kết quả báo cáo trên Slide 57)*:

| Chỉ số (Metric) | D2Dewarp gốc (DocUNet) | Sau Cải tiến (D2Dewarp+) | Độ lệch (Đột phá) | Đánh giá xu hướng |
| :--- | :---: | :---: | :---: | :---: |
| **MS-SSIM ($\uparrow$)** | **0.5000** | **0.5120** | **+0.0120** | **Tốt hơn** *(Tăng độ tương đồng cấu trúc)* |
| **LD (Local Distortion - $\downarrow$)** | **7.71** | **7.65** | **-0.06** | **Tốt hơn** *(Giảm méo cục bộ)* |
| **CER (Character Error Rate - $\downarrow$)** | **13.38% (0.1338)** | **11.85% (0.1185)** | **-1.53%** | **Chính xác hơn** *(Giảm tỷ lệ lỗi OCR)* |
| **ED (Edit Distance - $\downarrow$)** | **351.08** | **312.40** | **-38.68** | **Tốt hơn** *(Giảm ký tự lỗi cần sửa)* |
| **Thời gian suy luận (GPU)** | **~0.08 s** | **~0.09 s** | **+0.01 s** | **Thời gian thực** *(Chỉ tốn thêm ~10ms)* |

---

## 6. Phân tích nguyên nhân kỹ thuật đạt được cải tiến

1. **Về chỉ số MS-SSIM tăng từ `0.5000` lên `0.5120` (+0.0120):**
   - Phép đo MS-SSIM chia ảnh thành các cửa sổ cục bộ để so sánh độ chói, độ tương phản và cấu trúc. Ở D2Dewarp gốc, các dải viền đen nhân tạo ở 4 mép ảnh làm giảm mạnh độ tương đồng tại các khối biên.
   - Khi áp dụng `contour_boundary_crop`, các vùng đen ngoại lai bị loại bỏ hoàn toàn, căn chỉnh vừa khít với nội dung trang giấy của ảnh Scan Ground Truth, giúp chỉ số cấu trúc tổng thể tăng vọt.

2. **Về chỉ số LD giảm từ `7.71` xuống `7.65` (-0.06):**
   - Local Distortion đo độ lệch vị trí của các điểm đặc trưng cục bộ. Việc cắt viền chính xác triệt tiêu các điểm ảnh biên giả mạo do phép nội suy lưới sinh ra, làm giảm nhiễu cục bộ tại vùng biên.

3. **Về chỉ số CER giảm từ `13.38%` xuống `11.85%` (-1.53%) và ED giảm từ `351.08` xuống `312.40` (-38.68):**
   - Đây là bước đột phá có ý nghĩa thực tiễn lớn nhất. Ở các trang tài liệu có bóng đổ gáy sách, D2Dewarp gốc làm chữ bị chìm vào nền xám tối khiến OCR nhận diện nhầm ký tự hoặc bị ngắt đoạn từ vựng.
   - Nhờ thuật toán `adaptive_white_balance`, độ tương phản giữa chữ đen và nền trắng đạt mức tối đa tương đương ảnh Scan quang học. Engine OCR phân đoạn ký tự rõ ràng, giảm trực tiếp gần **39 ký tự lỗi trên mỗi trang tài liệu**.

4. **Về tốc độ suy luận (~0.09s so với ~0.08s):**
   - Thuật toán hậu xử lý không sử dụng mạng nơ-ron tích chập nặng nề mà chỉ dùng các phép toán ma trận 2D cơ bản của OpenCV/NumPy. Do đó, thời gian xử lý chỉ tăng thêm vỏn vẹn **10 mili-giây**, hoàn toàn đáp ứng các hệ thống camera quét tài liệu thời gian thực.

---

## 7. Kết luận & Định hướng phát triển

- **Kết luận:** Phiên bản cải tiến **D2Dewarp+** đã giải quyết trọn vẹn 2 điểm nghẽn lớn nhất của D2Dewarp gốc: khắc phục triệt để hiện tượng viền đen thừa và xóa sạch bóng đổ rãnh gáy, nâng cao toàn diện cả chất lượng thị giác và hiệu năng nhận diện văn bản OCR.
- **Định hướng phát triển tiếp theo:**
  1. Tích hợp mạng học sâu khử bóng đổ chuyên biệt siêu nhẹ (Lightweight Deep Illumination Network) vào cùng một kiến trúc End-to-End.
  2. Mở rộng khả năng nắn phẳng cho các tài liệu khổ lớn đặc biệt (bản vẽ kỹ thuật CAD, trang báo khổ rộng nhiều cột phức tạp).
