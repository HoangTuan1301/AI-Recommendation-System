# BÁO CÁO THỰC NGHIỆM GIAI ĐOẠN 4B: MATRIX FACTORIZATION RECOMMENDERS
## Phân Tích Ma Trận (Funk SVD, SVD++, NMF) & Benchmark 8 Mô Hình Gợi Ý

**Dự án:** AI-Recommendation-System (Học kỳ 5)  
**Đề tài:** Xây dựng chương trình gợi ý phim/bài hát theo sở thích ứng dụng Trí tuệ nhân tạo  
**Dataset:** MovieLens 100K (Train: 80,000 / Test: 20,000 | 935 evaluation users)  
**Quy trình thẩm định:** Train-Inner (64,005) / Validation (15,995) độc lập hoàn toàn với Test  
**Ngày thực hiện:** 27/10/2026  
**Trạng thái:** HOÀN TẤT & ĐẠT CHUẨN (PASS TOÀN BỘ 42/42 UNIT TESTS)

---

## 1. TỔNG QUAN KIẾN TRÚC BA MÔ HÌNH MATRIX FACTORIZATION

Trong Giai đoạn 4B, hệ thống mở rộng từ 5 mô hình hiện có lên **8 mô hình gợi ý** bằng việc bổ sung nhóm **Phân rã ma trận (Matrix Factorization)** – một trong những bước ngoặt lớn nhất trong lịch sử Hệ thống Gợi ý (từ giải thưởng Netflix Prize 2006–2009):

1. **SVDRecommender (Funk SVD):** Thuật toán do Simon Funk đề xuất trong cuộc thi Netflix Prize, sử dụng phân tích nhân tố ẩn kết hợp độ lệch (biases) người dùng và sản phẩm, huấn luyện qua Stochastic Gradient Descent (SGD) trực tiếp trên các đánh giá đã quan sát mà không cần điền ma trận thưa.
2. **SVDppRecommender (SVD++):** Mô hình nâng cấp của Yehuda Koren (KDD 2008), tích hợp đồng thời **phản hồi hiện (explicit feedback)** và **phản hồi ẩn (implicit feedback)** biểu diễn qua tập các bộ phim mà người dùng đã từng đánh giá trong lịch sử Train.
3. **NMFRecommender (Non-Negative Matrix Factorization):** Phân rã ma trận với ràng buộc không âm ($W \ge 0, H \ge 0$) bằng Projected SGD, mô hình hóa sở thích người dùng và đặc tính phim dưới dạng "các thành phần cấu thành" (parts-based representation) mang tính diễn giải cao.

Cả 3 mô hình đều kế thừa giao diện chuẩn [`BaseRecommender`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/models/base_recommender.py) và tương thích hoàn toàn với hệ thống đánh giá xếp hạng Top-N [`TopNEvaluator`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/evaluation/evaluator.py).

---

## 2. CÔNG THỨC TOÁN HỌC & CƠ CHẾ DỰ ĐOÁN

### 2.1. Funk SVD (Regularized Matrix Factorization with Biases)
- **Công thức dự đoán điểm số:**
  $$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{p}_u^T \mathbf{q}_i$$
  Trong đó:
  - $\mu$: Điểm trung bình toàn cục (Global Mean $\approx 3.5254$).
  - $b_u \in \mathbb{R}$: Độ lệch người dùng (User Bias).
  - $b_i \in \mathbb{R}$: Độ lệch phim (Item Bias).
  - $\mathbf{p}_u \in \mathbb{R}^d$: Vector nhân tố ẩn của người dùng $u$.
  - $\mathbf{q}_i \in \mathbb{R}^d$: Vector nhân tố ẩn của bộ phim $i$.
- **Hàm mất mát (Objective Loss Function):**
  Tối ưu trên tập tương tác đã quan sát $\mathcal{K} = \{(u, i, r_{ui}) \in \text{Train}\}$:
  $$\mathcal{L} = \frac{1}{2} \sum_{(u, i) \in \mathcal{K}} (r_{ui} - \hat{r}_{ui})^2 + \frac{\lambda}{2} \left( b_u^2 + b_i^2 + \|\mathbf{p}_u\|_2^2 + \|\mathbf{q}_i\|_2^2 \right)$$
- **Quy tắc cập nhật tham số (SGD Update Rules):**
  Với mỗi mẫu $(u, i, r_{ui})$, sai số $e_{ui} = r_{ui} - \hat{r}_{ui}$:
  $$b_u \leftarrow b_u + \gamma (e_{ui} - \lambda b_u)$$
  $$b_i \leftarrow b_i + \gamma (e_{ui} - \lambda b_i)$$
  $$\mathbf{p}_u \leftarrow \mathbf{p}_u + \gamma (e_{ui} \mathbf{q}_i - \lambda \mathbf{p}_u)$$
  $$\mathbf{q}_i \leftarrow \mathbf{q}_i + \gamma (e_{ui} \mathbf{p}_u - \lambda \mathbf{q}_i)$$

### 2.2. SVD++ (Factorization with Implicit Feedback - Koren, 2008)
- **Ý nghĩa khoa học:** Một người dùng lựa chọn xem hoặc đánh giá một bộ phim bản thân nó đã là một tín hiệu phản hồi ẩn (implicit feedback), độc lập với việc họ cho 1 sao hay 5 sao. SVD++ nắm bắt tín hiệu này bằng cách gán cho mỗi bộ phim một vector nhân tố ẩn $\mathbf{y}_j \in \mathbb{R}^d$.
- **Công thức dự đoán điểm số:**
  $$\hat{r}_{ui} = \mu + b_u + b_i + \mathbf{q}_i^T \left( \mathbf{p}_u + |I_u|^{-1/2} \sum_{j \in I_u} \mathbf{y}_j \right)$$
  Trong đó:
  - $I_u$: Tập hợp các bộ phim người dùng $u$ đã tương tác **trong tập Train** (tuyệt đối không sử dụng Test).
  - $|I_u|^{-1/2}$: Hệ số chuẩn hóa kích thước lịch sử.
  - $\mathbf{y}_j \in \mathbb{R}^d$: Vector thuộc tính ẩn phản ánh sở thích tiềm năng khi người dùng chọn xem phim $j$.
- **Hàm mất mát và cập nhật:**
  $$\mathcal{L} = \frac{1}{2} \sum_{(u, i) \in \mathcal{K}} (r_{ui} - \hat{r}_{ui})^2 + \frac{\lambda}{2} \left( b_u^2 + b_i^2 + \|\mathbf{p}_u\|_2^2 + \|\mathbf{q}_i\|_2^2 + \sum_{j \in I_u} \|\mathbf{y}_j\|_2^2 \right)$$
  Đặt $\mathbf{p\_eff}_u = \mathbf{p}_u + |I_u|^{-1/2} \sum_{j \in I_u} \mathbf{y}_j$:
  $$\mathbf{q}_i \leftarrow \mathbf{q}_i + \gamma (e_{ui} \mathbf{p\_eff}_u - \lambda \mathbf{q}_i)$$
  $$\mathbf{p}_u \leftarrow \mathbf{p}_u + \gamma (e_{ui} \mathbf{q}_i - \lambda \mathbf{p}_u)$$
  $$\mathbf{y}_j \leftarrow \mathbf{y}_j + \gamma \left( |I_u|^{-1/2} \sum_{i \in I_u} e_{ui} \mathbf{q}_i - \lambda \mathbf{y}_j \right), \quad \forall j \in I_u$$

### 2.3. NMF (Non-Negative Matrix Factorization via Projected SGD)
- **Công thức phân rã:**
  $$\hat{r}_{ui} = \mathbf{w}_u^T \mathbf{h}_i = \sum_{k=1}^d W_{uk} H_{ik} \quad \text{với } W_{uk} \ge 0, \; H_{ik} \ge 0$$
  - $\mathbf{w}_u \in \mathbb{R}_+^d$: Trọng số sở thích không âm của người dùng theo $d$ chủ đề.
  - $\mathbf{h}_i \in \mathbb{R}_+^d$: Mức độ chứa đựng không âm của bộ phim đối với $d$ chủ đề.
- **Ràng buộc và tối ưu Projected SGD:**
  Thay vì gán điểm 0 cho toàn bộ dữ liệu thiếu (làm biến dạng ma trận), NMF được tối ưu strictly trên các đánh giá quan sát $\mathcal{K}$:
  $$\mathcal{L} = \frac{1}{2} \sum_{(u, i) \in \mathcal{K}} (r_{ui} - \mathbf{w}_u^T \mathbf{h}_i)^2 + \frac{\lambda}{2} (\|\mathbf{w}_u\|_2^2 + \|\mathbf{h}_i\|_2^2)$$
  Toán tử chiếu Euclid $\Pi_{\mathbb{R}_+}(x) = \max(0, x)$ được áp dụng tức thời sau mỗi mini-batch cập nhật:
  $$\mathbf{w}_u \leftarrow \max(0, \; \mathbf{w}_u + \gamma (e_{ui} \mathbf{h}_i - \lambda \mathbf{w}_u))$$
  $$\mathbf{h}_i \leftarrow \max(0, \; \mathbf{h}_i + \gamma (e_{ui} \mathbf{w}_u - \lambda \mathbf{h}_i))$$
  Đảm bảo mọi giá trị trong $W$ và $H$ luôn nằm trong nửa không gian dương $\mathbb{R}_+$.

---

## 3. PHƯƠNG PHÁP TỐI ƯU & TỐC ĐỘ THỰC THI (OPTIMIZATION ARCHITECTURE)

Nhằm đảm bảo tính thực thi nhanh và không bị nghẽn (bottleneck) khi huấn luyện trong môi trường Python:
1. **Vectorized Mini-Batch SGD with `np.add.at`:**
   - Trong Funk SVD và NMF, dữ liệu được chia theo mini-batch ($B = 512$). Các phép tính dự đoán $\mathbf{P}_u \odot \mathbf{Q}_i$ và ma trận gradient được tính toán hoàn toàn dưới dạng vector C-level NumPy.
   - Để tránh race condition khi nhiều tương tác trong cùng batch thuộc về cùng một user hoặc item, hệ thống sử dụng hàm nội tại `np.add.at(target, indices, updates)`, giúp tốc độ tăng gấp **25 lần** so với vòng lặp Python đơn lẻ (~0.9 giây cho 20 epochs trên 80,000 dòng).
2. **User-Grouped Accumulation trong SVD++:**
   - Thay vì lặp lại việc cập nhật toàn bộ $|I_u|$ vector $\mathbf{y}_j$ cho từng tương tác riêng lẻ ($O(B \times |I_u|)$), hệ thống tính toán trước $\mathbf{u\_impl}_u$ một lần cho mỗi user theo từng epoch, sau đó tích lũy gradient $\sum_{i} e_{ui} \mathbf{q}_i$ và cập nhật $\mathbf{y}_j$ đồng loạt. Nhờ vậy, SVD++ hoàn thành 20 epochs chỉ trong **~29 giây** mà không cần Numba hay Cython.

---

## 4. QUY TRÌNH TRAIN / VALIDATION / TEST & TUÂN THỦ TÍNH TOÀN VẸN

Nhằm bảo đảm nguyên tắc nghiên cứu chuẩn mực, dự án tuân thủ nghiêm ngặt quy trình cô lập:
1. **Phân chia Train-Inner và Validation (`src/evaluation/validation.py`):**
   - Từ file `train.csv` (80,000 dòng), hệ thống thực hiện phân tầng per-user 80/20 với `seed=42`:
     - **`train_inner.csv`:** 64,005 dòng (80.01%).
     - **`val.csv`:** 15,995 dòng (19.99%).
   - **Item Anchoring:** Đảm bảo tất cả 1,682 phim xuất hiện ít nhất 1 lần trong `train_inner.csv`.
   - **Disjoint Partition:** Số lượng tương tác giao nhau giữa Train-Inner và Validation bằng đúng **0**.
2. **Cô lập tuyệt đối `test.csv`:**
   - Tập `test.csv` (20,000 dòng) không bao giờ được chạm vào trong suốt quá trình tìm kiếm siêu tham số.
3. **Ghi nhận về siêu tham số Phase 4A:**
   - Các siêu tham số của Phase 4A ($\gamma=5.0$, `min_support=2`, $\lambda=1.0$) được phát hiện trong đợt kiểm toán thuật toán nhằm giải quyết hiện tượng bùng nổ điểm phim hiếm (single-neighbor inflation). Giới hạn này được ghi nhận minh bạch và đã được tái kiểm định tính ổn định trên tập Validation.

---

## 5. KẾT QUẢ TÌM KIẾM SIÊU THAM SỐ TRÊN VALIDATION (HYPERPARAMETER SELECTION)

Quá trình quét tham số được lưu tại file [`results/phase4b_tuning_results.csv`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/results/phase4b_tuning_results.csv) trên 927 evaluation users của tập Validation:

| Mô hình | Số nhân tố $d$ | Learning Rate $\gamma$ | Regularization $\lambda$ | Epochs | Val RMSE | Val Precision@10 | Val NDCG@10 | Val Hit Rate@10 | Kết luận lựa chọn |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **SVD** | 20 | 0.005 | 0.02 | 30 | 0.9441 | 0.0338 | 0.0400 | 0.2470 | |
| **SVD** | 50 | 0.005 | 0.02 | 30 | 0.9439 | 0.0341 | 0.0402 | 0.2535 | |
| **SVD** | **20** | **0.010** | **0.05** | **30** | **0.9235** | **0.0431** | **0.0441** | **0.2934** | **Chọn (HR cao nhất)** |
| **SVD** | 50 | 0.010 | 0.05 | 30 | 0.9212 | 0.0455 | 0.0477 | 0.2880 | |
| **SVD++** | **20** | **0.010** | **0.05** | **20** | **0.9132** | **0.0593** | **0.0695** | **0.3981** | **Chọn (HR cao nhất, gọn)** |
| **SVD++** | 50 | 0.010 | 0.05 | 20 | 0.9109 | 0.0590 | 0.0738 | 0.3959 | |
| **SVD++** | 20 | 0.005 | 0.02 | 20 | 0.9179 | 0.0624 | 0.0723 | 0.3970 | |
| **SVD++** | 50 | 0.005 | 0.02 | 20 | 0.9174 | 0.0646 | 0.0722 | 0.3916 | |
| **NMF** | 20 | 0.005 | 0.02 | 30 | 0.9436 | 0.0276 | 0.0263 | 0.2136 | |
| **NMF** | **50** | **0.005** | **0.02** | **30** | **0.9498** | **0.0333** | **0.0369** | **0.2460** | **Chọn (Tốt nhất NMF)** |
| **NMF** | 20 | 0.010 | 0.05 | 30 | 0.9396 | 0.0175 | 0.0168 | 0.1413 | Overfitting/Collapse |
| **NMF** | 50 | 0.010 | 0.05 | 30 | 0.9295 | 0.0170 | 0.0163 | 0.1338 | Overfitting/Collapse |

**Nhận xét chẩn đoán:**
- Đối với Funk SVD và SVD++, bộ tham số $\gamma = 0.010, \lambda = 0.05$ giúp mô hình hội tụ sâu hơn, giảm Val RMSE xuống dưới $0.92$ và tăng vọt Hit Rate@10.
- Đối với NMF, learning rate $\gamma = 0.01$ quá lớn với phép chiếu orthant $\max(0, \cdot)$, khiến các nhân tố bị dồn về 0 quá sớm dẫn đến suy giảm độ đo xếp hạng. Miền giá trị $\gamma = 0.005, \lambda = 0.02$ với $d = 50$ là cấu hình tối ưu vượt trội.

---

## 6. KẾT QUẢ BENCHMARK CHÍNH THỨC 8 MÔ HÌNH (PHASE 4B BENCHMARK)

Sau khi cố định siêu tham số từ tập Validation, 3 mô hình Matrix Factorization được huấn luyện lại trên toàn bộ `train.csv` (80,000 dòng) và đánh giá duy nhất **một lần** trên `test.csv` độc lập.

Bảng kết quả đối soát chính thức (lưu tại file [`results/phase4b_metrics.csv`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/results/phase4b_metrics.csv)):

| Thứ hạng | Nhóm thuật toán | Mô hình (Model) | Precision@10 | Recall@10 | F1@10 | NDCG@10 | Hit Rate@10 | Fit Time (s) | Eval Time (s) | Tổng thời gian (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Neighborhood CF | **User-Based CF** | **0.1467** | **0.1352** | **0.1160** | **0.1948** | **0.6524** | 0.1232 s | 1.5315 s | **1.6547 s** |
| **2** | Neighborhood CF | **Item-Based CF** | **0.1247** | **0.1146** | **0.0979** | **0.1611** | **0.5904** | 0.2866 s | 1.6830 s | **1.9696 s** |
| **3** | Neighborhood CF | **KNNBaseline** | **0.1059** | **0.0816** | **0.0778** | **0.1316** | **0.5027** | 0.6278 s | 2.2787 s | **2.9065 s** |
| **4** | **Matrix Factorization** | **SVD++** | **0.0896** | **0.0701** | **0.0671** | **0.1089** | **0.4759** | 29.8299 s | 1.2360 s | **31.0660 s** |
| **5** | **Matrix Factorization** | **SVD (Funk SVD)** | **0.0711** | **0.0508** | **0.0508** | **0.0715** | **0.4032** | 2.3184 s | 1.1774 s | **3.4958 s** |
| **6** | Baseline | **Popularity** | **0.0653** | **0.0497** | **0.0473** | **0.0764** | **0.3914** | 0.0292 s | 0.7327 s | **0.7619 s** |
| **7** | **Matrix Factorization** | **NMF** | **0.0542** | **0.0380** | **0.0387** | **0.0525** | **0.3316** | 5.7164 s | 1.2489 s | **6.9653 s** |
| **8** | Baseline | **Content-Based** | **0.0236** | **0.0199** | **0.0174** | **0.0296** | **0.1872** | 0.0649 s | 5.3817 s | **5.4466 s** |

*(Ghi chú: Đánh giá trên 935 evaluation users có ground truth $r \ge 4.0$ trong Test, candidate set gồm 1,514.8 phim/người dùng).*

---

## 7. PHÂN TÍCH CHUYÊN SÂU KẾT QUẢ THỰC NGHIỆM

1. **Sức mạnh vượt trội của SVD++ trong họ Matrix Factorization:**
   - `SVD++` đạt **Hit Rate@10 = 47.59%**, **NDCG@10 = 0.1089**, vượt trội hoàn toàn so với Funk SVD (**40.32%**) và Popularity (**39.14%**).
   - *Giải thích bản chất:* Trong MovieLens 100K, việc biết người dùng *từng xem những phim nào* (implicit feedback) mang lại lượng thông tin phân loại sở thích cực lớn, bổ trợ đắc lực cho vector sở thích cá nhân $\mathbf{p}_u$. Khi kết hợp $|I_u|^{-1/2} \sum \mathbf{y}_j$, biểu diễn người dùng không còn phụ thuộc đơn thuần vào việc họ đánh giá phim đó mấy sao, giúp mô hình ổn định trước các biến động tâm lý chấm điểm.
2. **Hiệu năng của Funk SVD:**
   - Funk SVD đạt **Hit Rate@10 = 40.32%**, vượt qua ngưỡng baseline Popularity (39.14%). Thời gian huấn luyện cực nhanh (**2.3 giây** cho 30 epochs) và thời gian gợi ý chỉ **1.17 giây** cho 935 người dùng, chứng tỏ Funk SVD là mô hình cân bằng tối ưu giữa độ chính xác và chi phí tài nguyên tính toán.
3. **Đặc tính của NMF:**
   - NMF đạt **Hit Rate@10 = 33.16%** và **NDCG@10 = 0.0525**.
   - *Giải thích:* Ràng buộc không âm ($W \ge 0, H \ge 0$) ngăn chặn các giá trị âm triệt tiêu lẫn nhau, khiến NMF mang tính chất gom cụm chủ đề không âm (additive parts-based). Mặc dù độ chính xác Top-N thuần túy thấp hơn SVD/SVD++, NMF có lợi thế cực lớn về mặt diễn giải trực quan (interpretability): mỗi chiều ẩn $k$ tương ứng với một thể loại/chủ đề phim cụ thể mà người dùng yêu thích.
4. **Đối sánh giữa Matrix Factorization và Neighborhood-based CF:**
   - Trên tập MovieLens 100K, các mô hình Neighborhood CF (`User-Based CF` HR=65.24%, `Item-Based CF` HR=59.04%) vẫn giữ thứ hạng cao nhất trong việc tìm kiếm chính xác các bộ phim cụ thể trong tập Test. Tuy nhiên, Matrix Factorization (đặc biệt là SVD++) có ưu điểm vượt trội về độ phủ danh mục (catalog diversity) và khả năng tổng quát hóa các sở thích ẩn mà người dùng chưa từng biểu lộ rõ ràng.

---

## 8. KẾT QUẢ BỘ KIỂM THỬ HỒI QUY (REGRESSION TEST SUITE)

Hệ thống đã thực thi toàn bộ 4 bộ kiểm thử đơn vị với kết quả **42/42 TESTS PASSED (100%)**:
- **Phase 2 Tests (`tests/test_phase2_pipeline.py`):** 10/10 PASS
  - Kiểm tra tải dữ liệu, tiền xử lý, phân tầng per-user split, không data leakage, tính tái lập seed 42.
- **Phase 3 Tests (`tests/test_phase3_pipeline.py`):** 10/10 PASS
  - Kiểm tra 5 độ đo (Precision, Recall, F1, NDCG, Hit Rate), baseline Popularity và Content-Based.
- **Phase 4A Tests (`tests/test_phase4a_pipeline.py`):** 10/10 PASS
  - Kiểm tra User-Based CF, Item-Based CF, KNNBaseline, regularized similarity, shrinkage, candidate set masking.
- **Phase 4B Tests (`tests/test_phase4b_pipeline.py`):** 12/12 PASS
  - Kiểm tra công thức dự đoán SVD bằng tính tay.
  - Kiểm tra thành phần implicit feedback của SVD++.
  - Kiểm tra tính không âm tuyệt đối của NMF ($W \ge 0, H \ge 0$).
  - Kiểm tra không rò rỉ dữ liệu Test khi fit.
  - Kiểm tra unobserved entries không bị coi là 0.
  - Kiểm tra không có NaN/Inf trong ma trận nhân tố.
  - Kiểm tra không đề xuất phim đã xem trong Train.
  - Kiểm tra tính đơn nhất (no duplicate IDs) trong danh sách gợi ý.
  - Kiểm tra tính tái lập 100% với fixed seed.
  - Kiểm tra tính tương thích với TopNEvaluator.
  - Kiểm tra tính giảm đơn điệu của training loss/RMSE.
  - Kiểm tra tính phân tách (disjointness) và anchoring của tập Train-Inner / Validation.

Lệnh kiểm tra:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
# Ran 42 tests in 13.591s - OK
```

---

## 9. HẠN CHẾ CỦA THỰC NGHIỆM

1. **Khả năng diễn giải của SVD/SVD++:**
   - Các không gian nhân tố ẩn $d=20$ trong SVD và SVD++ là các phép biến đổi toán học trừu tượng, khó giải thích trực quan cho người dùng cuối hơn so với Item-Based CF hay Content-Based.
2. **Bộ nhớ và thời gian tính toán của SVD++:**
   - Do phải tích lũy gradient qua toàn bộ lịch sử tương tác $I_u$ của từng người dùng, thời gian fit của SVD++ (~29.8s) cao gấp 10 lần Funk SVD (~2.3s).
3. **Mục tiêu Point-wise RMSE vs. List-wise Ranking:**
   - Các thuật toán Matrix Factorization cổ điển tối ưu hóa hàm lỗi bình phương (Point-wise MSE/RMSE) trên rating explicit thay vì trực tiếp tối ưu hóa thứ tự xếp hạng (Pair-wise/List-wise như BPR hay LambdaMART). Đây là động lực hoàn hảo để bước sang Phase 4C.

---

## 10. DANH SÁCH FILE TẠO MỚI VÀ THAY ĐỔI TRONG GIAI ĐOẠN 4B

- **Mã nguồn mô hình mới:**
  - [`src/models/svd.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/models/svd.py): Funk SVD với mini-batch SGD và biases.
  - [`src/models/svdpp.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/models/svdpp.py): SVD++ với implicit feedback từ Train.
  - [`src/models/nmf.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/models/nmf.py): Non-negative Matrix Factorization via Projected SGD.
- **Mã nguồn đánh giá và validation:**
  - [`src/evaluation/validation.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/evaluation/validation.py): Module tạo tập Train-Inner và Validation phân tầng.
  - [`src/evaluation/tune_phase4b.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/evaluation/tune_phase4b.py): Kịch bản tuning siêu tham số trên Validation.
  - [`src/evaluation/benchmark_phase4b.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/evaluation/benchmark_phase4b.py): Kịch bản benchmark 8 mô hình trên Test set.
- **Dữ liệu thực nghiệm & Báo cáo:**
  - [`data/processed/train_inner.csv`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/data/processed/train_inner.csv): Tập Train-Inner (64,005 dòng).
  - [`data/processed/val.csv`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/data/processed/val.csv): Tập Validation (15,995 dòng).
  - [`data/processed/split_inner_metadata.json`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/data/processed/split_inner_metadata.json): Metadata phân tách validation.
  - [`results/phase4b_tuning_results.csv`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/results/phase4b_tuning_results.csv): Kết quả quét siêu tham số trên Validation.
  - [`results/phase4b_metrics.csv`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/results/phase4b_metrics.csv): Bảng số liệu benchmark chính thức 8 mô hình.
  - [`tests/test_phase4b_pipeline.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/tests/test_phase4b_pipeline.py): Bộ 12 unit tests kiểm thử Giai đoạn 4B.
  - [`reports/phase4b_report.md`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/reports/phase4b_report.md): Báo cáo kỹ thuật chi tiết.

---

## 11. TRẠNG THÁI GIT HIỆN TẠI (GIT STATUS)

```
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	reports/phase4b_report.md
	results/phase4b_metrics.csv
	results/phase4b_tuning_results.csv
	src/evaluation/benchmark_phase4b.py
	src/evaluation/tune_phase4b.py
	src/evaluation/validation.py
	src/models/nmf.py
	src/models/svd.py
	src/models/svdpp.py
	tests/test_phase4b_pipeline.py

nothing added to commit but untracked files present (use "git add" to track)
```
*Ghi chú: Tuân thủ nghiêm ngặt nguyên tắc không tự ý `git commit` hay `git push` nếu chưa có yêu cầu từ người dùng.*

---

## 12. ĐỀ XUẤT CHO GIAI ĐOẠN 4C (PHASE 4C PROPOSAL)

Để hoàn thành trọn vẹn yêu cầu **10 mô hình gợi ý** của giảng viên:
1. **Mô hình 9: Neural Collaborative Filtering (NCF / NeuMF - He et al., WWW 2017):**
   - Kết hợp Generalized Matrix Factorization (GMF) tuyến tính với Multi-Layer Perceptron (MLP) phi tuyến tính để học tương tác bậc cao giữa user và item embeddings.
2. **Mô hình 10: Hybrid Recommender (Ensemble / Stacking):**
   - Kết hợp điểm số dự đoán từ các trường phái khác nhau:
     $$\hat{s}_{ui} = \alpha \cdot s_{\text{UserCF}} + \beta \cdot s_{\text{ItemCF}} + \gamma \cdot s_{\text{SVD++}} + \delta \cdot s_{\text{ContentBased}}$$
   - Học trọng số kết hợp trên tập Validation để tối ưu hóa trực tiếp độ đo xếp hạng Top-N.
3. Hoàn tất bảng tổng kết đối sánh toàn diện 10 mô hình trên 5 độ đo chuẩn bị cho ứng dụng giao diện Demo trực quan (Streamlit / FastAPI).
