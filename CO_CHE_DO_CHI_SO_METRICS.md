# CƠ CHẾ HOẠT ĐỘNG VÀ PHƯƠNG PHÁP ĐO CÁC CHỈ SỐ BENCHMARK (MS-SSIM, CER, ED)

Tài liệu này giải thích chi tiết toàn bộ cơ chế, công thức toán học, thuật toán và quy trình thực nghiệm được cài đặt trong file [run_benchmark_comparison.py](run_benchmark_comparison.py) nhằm đánh giá định lượng chất lượng của các mô hình nắn phẳng tài liệu (**Document Dewarping**).

---

## 1. Tổng quan mục đích đánh giá

Trong bài toán nắn phẳng tài liệu chụp cong (DocDewarp), việc đánh giá mắt thường (định tính) là chưa đủ. Để chứng minh tính ưu việt của mô hình một cách học thuật, hệ thống đánh giá trên 2 khía cạnh độc lập:

1. **Khía cạnh Hình học & Thị giác (Geometric & Visual Quality):** Đo lường mức độ khôi phục hình dạng tổng thể của trang tài liệu so với bản quét phẳng thực tế (**Ground Truth Scan**).
2. **Khía cạnh Nhận dạng Văn bản Thực tế (Downstream OCR Usability):** Đo lường độ méo mó của từng ký tự bằng cách xem phần mềm OCR (nhận dạng ký tự quang học) có thể đọc chính xác văn bản sau nắn phẳng hay không.

```
                   ┌──────────────────────────────────────────┐
                   │ Ảnh tài liệu bị uốn cong (Distorted)     │
                   └────────────────────┬─────────────────────┘
                                        │
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │ Mô hình Dewarping (D2Dewarp / DocTr)     │
                   └────────────────────┬─────────────────────┘
                                        │
                         Ảnh nắn phẳng (Dewarped Image)
                                        │
          ┌─────────────────────────────┴─────────────────────────────┐
          ▼                                                           ▼
┌───────────────────────────────────┐               ┌───────────────────────────────────┐
│     ĐÁNH GIÁ THỊ GIÁC & HÌNH HỌC   │               │       ĐÁNH GIÁ KÝ TỰ VĂN BẢN      │
│               MS-SSIM             │               │             ED & CER              │
│  (Multi-Scale Structural Sim.)    │               │  (Levenshtein Distance via OCR)   │
└───────────────────────────────────┘               └───────────────────────────────────┘
```

---

## 2. Giải thích chi tiết: Làm sao để đo được từng chỉ số?

### 2.1. Chỉ số MS-SSIM (Multi-Scale Structural Similarity Index)

#### Vì sao không dùng MSE hoặc PSNR thông thường?
- **MSE (Mean Squared Error)** và **PSNR (Peak Signal-to-Noise Ratio)** so sánh từng điểm ảnh $pixel(x, y)$ tại cùng một tọa độ tuyệt đối.
- Nếu trang giấy đã được nắn rất phẳng nhưng bị **lệch nhẹ 3–5 pixel** sang trái do căn lề, MSE sẽ tăng vọt và PSNR sẽ rất tệ, dù về mặt thị giác trang giấy hoàn hảo.
- **SSIM** khắc phục triệt để nhược điểm này bằng cách so sánh **cấu trúc không gian, độ sáng và độ tương phản cục bộ** thay vì giá trị điểm ảnh độc lập.

#### Quy trình đo đạc trong mã nguồn:
1. **Chuyển về ảnh xám (Grayscale):**
   ```python
   img_p = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)
   img_g = cv2.imread(gt_img_path, cv2.IMREAD_GRAYSCALE)
   ```
   Loại bỏ nhiễu màu sắc ánh sáng để tập trung hoàn toàn vào cấu trúc hình học của trang sách.

2. **Chuẩn hóa tỷ lệ theo tiêu chuẩn quốc tế DocUNet A4:**
   ```python
   p_res = cv2.resize(img_p, (595, 842))
   g_res = cv2.resize(img_g, (595, 842))
   ```
   Ảnh quét thực tế có thể có độ phân giải $2400 \times 3200$, còn ảnh đầu ra của mô hình có thể là $1200 \times 1600$. Để so sánh công bằng, cả 2 ảnh bắt buộc phải được đưa về kích thước chuẩn của khổ giấy A4 là **$595 \times 842$ pixels**.

3. **Công thức toán học tính SSIM:**
   Sử dụng hàm `skimage.metrics.structural_similarity`:
   $$SSIM(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
   Trong đó:
   - $\mu_x, \mu_y$: Cường độ sáng trung bình của ảnh dự đoán và ảnh scan chuẩn.
   - $\sigma_x^2, \sigma_y^2$: Phương sai thể hiện độ tương phản.
   - $\sigma_{xy}$: Hiệp phương sai thể hiện mức độ tương quan về mặt cấu trúc giữa 2 ảnh.
   - $C_1 = (K_1 L)^2, C_2 = (K_2 L)^2$: Các hằng số tránh lỗi chia cho 0 với dải động $L = 255$.

4. **Ý nghĩa số liệu:**
   - Dao động từ **$0.0$ đến $1.0$**.
   - **Càng cao càng tốt**. Điểm số càng gần $1.0$ thể hiện tài liệu nắn phẳng có cấu trúc, bố cục dòng kẻ và độ tương phản tương đồng cao nhất với bản quét gốc.

---

### 2.2. Chỉ số ED (Edit Distance - Khoảng cách Levenshtein)

#### Bản chất và mục đích:
- Đánh giá khả năng hỗ trợ OCR: Khi trang giấy còn bị cong vồng hoặc gập gãy, các chữ cái sẽ bị co rúm, nghiêng méo, khiến OCR nhận diện sai (ví dụ: chữ `c` bị nhận thành chữ `o`, chữ `rn` thành `m`, hoặc mất hẳn cả dòng chữ ở gáy sách).
- **Edit Distance** đo số lượng thao tác chỉnh sửa tối thiểu (Chèn, Xóa, hoặc Thay thế 1 ký tự) để biến chuỗi văn bản nhận dạng được từ ảnh nắn phẳng thành chuỗi văn bản của bản quét chuẩn.

#### Quy trình đo đạc trong mã nguồn:
1. **Trích xuất văn bản qua OCR:**
   File sử dụng thư viện `rapidocr_onnxruntime` (mô hình OCR chạy nhanh bằng ONNX Runtime) để quét text từ cả 2 ảnh:
   ```python
   text_pred = get_text(pred_img_path)
   text_gt = get_text(gt_img_path)
   ```

2. **Làm sạch chuỗi (Whitespace Normalization):**
   ```python
   gt_clean = "".join(text_gt.split())
   pred_clean = "".join(text_pred.split())
   ```
   *Điểm quan trọng:* Cần xóa toàn bộ khoảng trắng, dấu ngắt dòng `\n` và thụt lề `\t`. Điều này đảm bảo thuật toán chỉ đánh giá đúng **nội dung chữ**, không phạt mô hình vì việc ngắt dòng sớm/muộn hơn bản chuẩn.

3. **Tính khoảng cách Levenshtein:**
   ```python
   from rapidfuzz.distance import Levenshtein
   ed = Levenshtein.distance(gt_clean, pred_clean)
   ```
   Thuật toán quy hoạch động tính ma trận khoảng cách:
   $$D(i, j) = \begin{cases} \max(i, j) & \text{nếu } \min(i, j) = 0 \\ \min \begin{cases} D(i-1, j) + 1 & \text{(Xóa ký tự)} \\ D(i, j-1) + 1 & \text{(Chèn ký tự)} \\ D(i-1, j-1) + \mathbb{I}(s_1[i] \neq s_2[j]) & \text{(Thay thế ký tự)} \end{cases} & \text{khác} \end{cases}$$

4. **Ý nghĩa số liệu:**
   - Là một số nguyên $\ge 0$.
   - **Càng thấp càng tốt** (Lý tưởng nhất là bằng $0$ — văn bản nhận diện chính xác 100% từng ký tự).

---

### 2.3. Chỉ số CER (Character Error Rate - Tỉ lệ lỗi ký tự)

#### Vì sao cần có CER khi đã có ED?
- Nếu trang tài liệu ngắn có 200 ký tự mà sai 50 ký tự $\to$ Tỉ lệ lỗi rất nặng ($25\%$).
- Nếu một bài báo khoa học dài có 4000 ký tự mà sai 50 ký tự $\to$ Tỉ lệ lỗi rất thấp ($1.25\%$).
- Do đó, **CER chuẩn hóa khoảng cách ED theo tổng độ dài văn bản chuẩn**, cho phép so sánh công bằng giữa các trang sách có lượng chữ khác nhau.

#### Công thức toán học:
$$CER = \frac{\text{Edit Distance}(T'_{\text{pred}}, T'_{\text{gt}})}{\max(1, \text{len}(T'_{\text{gt}}))}$$

Cài đặt trong Python:
```python
cer = ed / max(1, len(gt_clean))
```

#### Ý nghĩa số liệu:
- Dao động thường từ **$0.0$ đến $1.0$** (có thể $> 1.0$ nếu mô hình sinh ra quá nhiều ký tự rác/nhiễu).
- **Càng thấp càng tốt**.
- Ví dụ: $CER = 0.052$ nghĩa là mô hình chỉ gặp lỗi $5.2\%$ tổng số ký tự của trang tài liệu.

---

## 3. Kiến trúc luồng thực thi trong `run_benchmark_comparison.py`

Pipeline tự động hóa gồm 5 bước được lặp qua từng mẫu kiểm thử:

```
[Đọc mẫu Benchmark] 
       │
       ▼
[Bước 1: Nạp ảnh & Tính Sobel Gradient] 
  └── Ghép 3 kênh RGB + 1 kênh Gradient thành Tensor 4 kênh (448x448x4).
       │
       ▼
[Bước 2: Dự đoán D2Dewarp & Tạo ảnh phẳng]
  ├── Mô hình dự đoán ma trận dòng ngang pred_h, dòng dọc pred_v và lưới tọa độ pred_bm.
  ├── Chuẩn hóa tọa độ: bm = (2.0 * (pred_bm / 448) - 1) * 1.004
  ├── Làm mờ cv2.blur((3, 3)) khử răng cưa.
  └── F.grid_sample(...) kéo điểm ảnh về vị trí phẳng ở kích thước gốc.
       │
       ▼
[Bước 3: Đo lường đồng thời 4 phương pháp]
  ├── (a) Distorted: Đo crop_path với scan_path
  ├── (b) Baseline 0: Đo page-dewarp với scan_path
  ├── (c) Baseline 1: Đo DocTr-Plus với scan_path
  └── (d) D2Dewarp: Đo d2_out_path với scan_path
       │
       ▼
[Bước 4: Sinh ảnh trực quan hóa]
  ├── Ghép 5 ô so sánh: [Distorted | Baseline 0 | DocTr | D2Dewarp | Scan GT]
  ├── Ghép trực quan dòng kép: [Input | H-Lines | V-Lines | Dewarped]
  └── Vẽ Header hiển thị trực tiếp điểm số SSIM và CER lên từng ảnh.
       │
       ▼
[Bước 5: Xuất kết quả]
  ├── Xuất file JSON: benchmark_results/benchmark_comparison_full.json
  └── In bảng tổng hợp so sánh ra màn hình Terminal.
```

---

## 4. Tối ưu hiệu năng: Cơ chế `text_cache`

Do quá trình nhận dạng ký tự bằng OCR trên CPU tốn từ 2 đến 4 giây mỗi ảnh, việc chạy lại OCR nhiều lần cho cùng một ảnh quét chuẩn (Ground Truth) sẽ làm chậm chương trình gấp nhiều lần.

Mã nguồn được cài đặt cơ chế **bộ nhớ đệm (In-memory Cache)**:
```python
text_cache = {}

def get_text(img_path):
    if not os.path.exists(img_path):
        return ""
    if img_path in text_cache:
        return text_cache[img_path]  # Trả về ngay lập tức nếu đã từng nhận dạng
    
    # Chỉ chạy OCR nếu file này chưa từng được đọc
    img = np.array(Image.open(img_path).convert('RGB'))
    res, _ = ocr(img)
    txt = " ".join([line[1] for line in res]) if res else ""
    text_cache[img_path] = txt
    return txt
```
Nhờ cơ chế này, ảnh Scan chuẩn chỉ bị quét OCR duy nhất 1 lần thay vì 4 lần cho mỗi mẫu dữ liệu.

---

## 5. Hướng dẫn chạy và đọc kết quả đo

### 5.1. Câu lệnh thực thi

Để đo các chỉ số trên tập benchmark có sẵn:
```bash
py run_benchmark_comparison.py
```

Tuỳ chỉnh đường dẫn dữ liệu nếu cần:
```bash
py run_benchmark_comparison.py --crop_dir "path/to/crop" --scan_dir "path/to/scan" --save_path "path/to/save"
```

### 5.2. File kết quả sinh ra
Sau khi chạy xong, toàn bộ kết quả sẽ được lưu trong thư mục `benchmark_results/`:
- **`benchmark_results/dewarp/`**: Chứa các ảnh nắn phẳng được xuất ra.
- **`benchmark_results/visualizations/comparison_5way_{id}.png`**: Ảnh trực quan ghép 5 cột kèm điểm SSIM và CER trên Header.
- **`benchmark_results/visualizations/dual_dimensions_vis_{id}.png`**: Ảnh biểu diễn lưới dòng ngang $F_h$ và dòng dọc $F_v$.
- **`benchmark_results/benchmark_comparison_full.json`**: Chứa toàn bộ số liệu chi tiết (độ dài ký tự, thời gian nắn phẳng, điểm SSIM, CER, ED của từng mẫu).

### 5.3. Mẫu bảng kết quả in ra màn hình
```text
Sample   | Category       | Method                 | MS-SSIM  | CER      | ED      | Chars        | Time (s)
-------------------------------------------------------------------------------------------------------------------
51_1     | Dễ - Vừa       | Distorted              | 0.4480   | 0.2850   | 245     | 614/859      | —       
         |                | Baseline 0 (page-dew.) | 0.4720   | 0.2210   | 190     | 669/859      | ~12.5   
         |                | Baseline 1 (DocTr)     | 0.5180   | 0.1650   | 142     | 717/859      | ~2.1    
         |                | D2Dewarp (Đề xuất)     | 0.5620   | 0.0820   | 70      | 789/859      | 0.412   
-------------------------------------------------------------------------------------------------------------------
```

---

## 6. Mẹo phân tích chuyên sâu khi bảo vệ báo cáo

1. **Có trường hợp nào SSIM cao nhưng CER lại kém không?**
   - **Có.** Ví dụ một thuật toán làm mờ (smooth) trang giấy quá mức: ảnh trông có vẻ phẳng và nền rất sạch nên SSIM tăng cao, nhưng chữ bị nhòe nét khiến bộ đọc OCR không nhận ra chữ $\to$ CER tăng đột biến.
   - Vì vậy, một mô hình được coi là xuất sắc **bắt buộc phải cải thiện đồng thời cả 2 chỉ số**: vừa đạt SSIM cao (hình học phẳng), vừa đạt CER thấp (chữ nét, không bị co kéo).

2. **Tại sao D2Dewarp / D2Dewarp+ đạt chỉ số CER vượt trội so với DocTr?**
   - DocTr chỉ dự đoán ma trận dòng 1 chiều (theo dòng quét dọc), nên khi gặp các tài liệu có cấu trúc 2 cột hoặc bảng biểu (như mẫu `48_2`), lưới tọa độ bị xô lệch ngang, làm đứt gãy chữ.
   - D2Dewarp sử dụng **ràng buộc 2 chiều đồng thời** ($F_h$ định hướng dòng chữ, $F_v$ định hướng lề và cột), giữ nguyên tỷ lệ khung hình của từng con chữ, giúp engine OCR đọc chính xác hầu như toàn bộ nội dung.
