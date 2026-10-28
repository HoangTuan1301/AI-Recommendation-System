# GIAO THỨC ĐÁNH GIÁ VÀ ĐÓNG BĂNG CHUẨN THỬ NGHIỆM — MUSICAI PHASE M3
*(Frozen Scientific Evaluation Protocol & Benchmark Pre-Registration)*

**Dự án:** Multi-Domain AI Recommendation System (CineAI V2.0 & MusicAI)  
**Tài liệu:** `docs/musicai/phase_m3_evaluation_protocol.md`  
**Giai đoạn:** Phase M3 — Offline Test Benchmark & Scientific Analysis  
**Trạng thái giao thức:** **FROZEN / PRE-REGISTERED (ĐÃ ĐÓNG BĂNG)**  
**Kỹ sư phụ trách:** Senior Recommender Systems Researcher & ML Quality Auditor  

---

## 1. MỤC TIÊU VÀ NGUYÊN TẮC BẢO TOÀN TÍNH KHOA HỌC

Tài liệu này xác lập và **đóng băng (freeze)** toàn bộ các thông số, phương pháp luận và quy tắc đánh giá trước khi thực thi giải mã và tính toán độ đo trên tập kiểm thử độc lập (`test.csv`). 

### Các nguyên tắc bất khả xâm phạm:
1. **Không điều chỉnh mô hình theo tập Test (Zero Test-tuning):** Tập Test chỉ dùng để đo lường năng lực tổng quát hóa cuối cùng (Final Generalization Performance). Tuyệt đối không tinh chỉnh siêu tham số, không sửa trọng số hoặc thay đổi checkpoint sau khi quan sát kết quả Test.
2. **Tiêu chuẩn chọn mô hình đã đóng băng từ Validation:** Mô hình đề xuất cho Backend được quyết định dựa trên bảng xếp hạng Validation tại Phase M2 (NDCG@10 và HR@10 kết hợp với độ trễ suy luận). Kết quả Test đóng vai trò nghiệm thu khoa học, không được dùng để "chọn lại" mô hình nếu thứ hạng thay đổi nhẹ.
3. **Đánh giá trên toàn bộ catalog ứng viên (Full-Catalog Ranking):** Không sử dụng kỹ thuật lấy mẫu 100 negatives (sampled negatives) nhằm bảo đảm tính khách quan và tránh thổi phồng kết quả.

---

## 2. ĐÓNG BĂNG CHECKPOINTS VÀ DẤU VÂN TAY DỮ LIỆU (FINGERPRINTS)

### 2.1. Danh sách 6 checkpoint đã đóng băng
Cả 6 checkpoint đã được niêm phong từ Phase M2 tại thư mục `models/checkpoints/music/`:

| STT | Mã mô hình | Tên mô hình | Đường dẫn tệp Checkpoint | Kích thước (Bytes) |
|:---:|---|---|---|:---:|
| 1 | `popularity` | Popularity Baseline | `models/checkpoints/music/popularity.pkl` | 1,212,666 |
| 2 | `item_cf` | Item-Based CF | `models/checkpoints/music/item_cf.pkl` | 4,130,649 |
| 3 | `user_cf` | User-Based CF | `models/checkpoints/music/user_cf.pkl` | 3,164,283 |
| 4 | `ials` | Implicit ALS (iALS) | `models/checkpoints/music/ials.pkl` | 1,922,394 |
| 5 | `bpr_mf` | BPR-MF | `models/checkpoints/music/bpr_mf.pkl` | 1,922,592 |
| 6 | `implicit_neural_cf` | Implicit Neural CF (NeuMF) | `models/checkpoints/music/implicit_neumf.pkl` | 2,634,021 |

### 2.2. Dấu vân tay dữ liệu (SHA-256 Fingerprints)
Các tệp dữ liệu được đóng băng tại `data/music/processed/` với mã SHA-256 xác thực từ `dataset_manifest.json`:
- **Tập Train (`train.csv`):** `d687485ab61b7c1e55aa162799c9cb4427e193fb68dca617329d73eb8b0728c3`
- **Tập Validation (`validation.csv`):** `badcbe04038f4b1cd7728fbe54b4ea0956b9c9e830eec86751c14457c32bf2cb`
- **Tập Test (`test.csv`):** `894cac7929dbbff491d92bfd2400938ff5c276329c2ea2c99b9cf98b1115b810`
- **Tập Metadata bài hát (`songs.csv`):** `53077d852a0cd4703a58e8ea6f8ef950b7ec99450c262c589a19c629fb66f8be`

---

## 3. KHÔNG GIAN ỨNG VIÊN VÀ QUY TẮC SUY LUẬN

1. **Tổng số người dùng đánh giá ($U_{test}$):** $2,520$ người dùng (toàn bộ người dùng trong `test.csv`).
2. **Tổng số bài hát trong danh mục ($I_{catalog}$):** $2,706$ bài hát.
3. **Quy tắc tạo tập ứng viên (Candidate Universe):**
   Với mỗi người dùng $u \in U_{test}$:
   $$\mathcal{C}_u = \mathcal{I}_{\text{catalog}} \setminus \mathcal{I}_u^{\text{train}}$$
   - $\mathcal{I}_u^{\text{train}}$: Tập hợp các bài hát người dùng $u$ đã từng tương tác trong tập Train (`train.csv`).
   - Mọi bài hát đã nghe trong Train đều bị **loại trừ bắt buộc** (`exclude_seen=True`).
   - Không gian ứng viên cho mỗi người dùng có quy mô: $|\mathcal{C}_u| \approx 2,706 - |\mathcal{I}_u^{\text{train}}|$ (trung bình ~2,691 bài hát chưa nghe).
4. **Quy tắc xếp hạng và xử lý điểm số bằng nhau (Tie-breaking):**
   - Các ứng viên được sắp xếp giảm dần theo điểm số dự đoán $\hat{s}(u, i)$.
   - Nếu xảy ra trường hợp bằng điểm số (ties), thuật toán sử dụng bộ sắp xếp ổn định (stable sort) dựa trên thứ tự index xuất hiện trước trong danh mục, bảo đảm tính xác thực và tái lập 100%.
   - Lấy đúng Top-$K$ với $K=10$.

---

## 4. ĐỊNH NGHĨA TOÁN HỌC CỦA 5 CHỈ SỐ ĐÁNH GIÁ TOP-10

Được tính theo phương pháp trung bình vĩ mô (Macro-Averaging) trên toàn bộ $|U_{test}| = 2,520$ người dùng.  
Gọi $R_u@K = [s_1, s_2, \dots, s_K]$ là danh sách $K=10$ bài hát được gợi ý cho người dùng $u$.  
Gọi $V_u \subset \mathcal{I}$ là tập hợp các bài hát thực tế của người dùng $u$ trong tập Test ($|V_u| \ge 1$).

### 4.1. Hit Rate@10 (HR@10)
Tỷ lệ người dùng có ít nhất một bài hát gợi ý trúng đích trong Top-10:
$$\text{HR}@10 = \frac{1}{|U|} \sum_{u \in U} \mathbb{I}\left( |R_u@10 \cap V_u| > 0 \right)$$

### 4.2. Precision@10 (P@10)
Tỷ lệ các bài hát được gợi ý trúng đích trên tổng số 10 bài hát gợi ý:
$$\text{Precision}@10 = \frac{1}{|U|} \sum_{u \in U} \frac{|R_u@10 \cap V_u|}{10}$$

### 4.3. Recall@10 (R@10)
Tỷ lệ các bài hát trong tập Test của người dùng được mô hình tìm thấy trong Top-10:
$$\text{Recall}@10 = \frac{1}{|U|} \sum_{u \in U} \frac{|R_u@10 \cap V_u|}{|V_u|}$$

### 4.4. Normalized Discounted Cumulative Gain@10 (NDCG@10)
Độ đo chất lượng xếp hạng có tính đến vị trí xuất hiện của bài hát trúng đích:
$$\text{DCG}_u@10 = \sum_{i=1}^{10} \frac{\mathbb{I}(s_i \in V_u)}{\log_2(i + 1)}$$
$$\text{IDCG}_u@10 = \sum_{i=1}^{\min(10, |V_u|)} \frac{1}{\log_2(i + 1)}$$
$$\text{NDCG}@10 = \frac{1}{|U|} \sum_{u \in U} \frac{\text{DCG}_u@10}{\text{IDCG}_u@10}$$

### 4.5. Mean Reciprocal Rank@10 (MRR@10)
Nghịch đảo vị trí của bài hát trúng đích đầu tiên trong Top-10:
$$\text{RR}_u@10 = \begin{cases} \frac{1}{\min \{i \in \{1,\dots,10\} \mid s_i \in V_u\}} & \text{nếu } |R_u@10 \cap V_u| > 0 \\ 0 & \text{ngược lại} \end{cases}$$
$$\text{MRR}@10 = \frac{1}{|U|} \sum_{u \in U} \text{RR}_u@10$$

---

## 5. CÁC ĐỘ ĐO BỔ SUNG CHO PHÂN TÍCH KHOA HỌC

1. **Độ phủ danh mục (Catalog Coverage):**
   $$\text{Coverage}@10 = \frac{\left| \bigcup_{u \in U} R_u@10 \right|}{|\mathcal{I}_{\text{catalog}}|}$$
2. **Chỉ số bất bình đẳng Gini (Gini Index of Recommendation Frequency):**
   Đo mức độ tập trung / thiên vị độ phổ biến của các gợi ý. Gini gần 0 biểu thị danh mục được gợi ý đồng đều; Gini gần 1 biểu thị gợi ý chỉ dồn vào một vài bài hát quen thuộc.
3. **Phân tầng người dùng theo mức độ tương tác (User Activity Stratification):**
   - **Tầng Thấp (Low-activity):** Người dùng có $\le 10$ bài hát trong Train.
   - **Tầng Trung bình (Medium-activity):** Người dùng có $11 - 25$ bài hát trong Train.
   - **Tầng Cao (High-activity):** Người dùng có $> 25$ bài hát trong Train.
4. **Khoảng tin cậy Bootstrap 95% (95% Bootstrap Confidence Intervals):**
   Lấy mẫu lại có hoàn lại theo cấp người dùng ($B = 1,000$ lần lặp) để thiết lập khoảng tin cậy cho NDCG@10 và HR@10.
5. **Độ trễ suy luận (Inference Latency):**
   Đo thời gian trung bình (ms/user) và thông lượng (users/giây) khi thực hiện gợi ý Top-10.

---

## 6. QUY TẮC LỰA CHỌN MÔ HÌNH CHO BACKEND FASTAPI

Tiêu chí lựa chọn mô hình đã được xác lập từ kết quả Validation Phase M2:
1. **Tiêu chí chất lượng gợi ý:** Điểm số NDCG@10 và HR@10 cao nhất trên tập Validation.
2. **Tiêu chí vận hành kỹ thuật:**
   - Thời gian suy luận $< 10$ ms/request.
   - Dung lượng checkpoint nạp vào bộ nhớ $< 10$ MB.
   - Không đòi hỏi bộ tăng tốc GPU khi phục vụ backend CPU.
3. **Mô hình dẫn đầu kỳ vọng:** **Item-Based CF** (đạt HR@10 = 0.2425, NDCG@10 = 0.1038 trên Validation, độ trễ suy luận cực thấp).
4. **Kiến trúc phục vụ đa mô hình (Unified Model Registry):** Tại Phase M4, hệ thống sẽ đăng ký cả 6 mô hình vào Factory để cho phép người dùng tùy chọn thuật toán trên giao diện người dùng.
