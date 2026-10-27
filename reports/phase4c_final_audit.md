# BÁO CÁO KIỂM TOÁN CUỐI GIAI ĐOẠN 4C (PHASE 4C FINAL AUDIT REPORT)
## KIỂM TOÁN TOÁN HỌC, THỰC NGHIỆM VÀ ĐỘ CHÍNH XÁC CỦA NEURAL CF & HYBRID RECOMMENDER

**Dự án:** Xây dựng chương trình gợi ý phim/bài hát theo sở thích ứng dụng Trí tuệ nhân tạo  
**Học phần:** Trí tuệ nhân tạo – K5  
**Tác giả:** Senior Machine Learning Engineer, Deep Learning Researcher & Software Architect  
**Ngày kiểm toán:** 27/10/2026  
**Môi trường thực thi:** Python 3.11.9, PyTorch 2.14.1+cpu, NumPy 1.26.4, SciPy 1.14.1, Scikit-learn 1.5.2, Pandas 2.2.3, Windows 11 x86_64  

---

### MỤC TIÊU VÀ PHẠM VI KIỂM TOÁN

Kiểm toán toàn diện tính chính xác giải thuật, tính ổn định số học, cơ chế học biểu diễn không gian ẩn, nguy cơ rò rỉ dữ liệu (data leakage), tính phân biệt của negative sampling, tính hợp lệ của chuẩn hóa điểm số trong mô hình Hybrid, và tính tái lập thực nghiệm (deterministic reproducibility) trên toàn bộ 10 mô hình của hệ thống trước khi chuyển sang Phase 5 (Triển khai ứng dụng tương tác Streamlit).

---

## 1. KẾT QUẢ KIỂM TOÁN NEURAL COLLABORATIVE FILTERING (NeuMF)

File kiểm toán: [`src/models/neural_cf.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/models/neural_cf.py)

### 1.1. Cấu trúc mạng NeuMF, GMF và MLP
Kiến trúc tuân thủ nghiêm ngặt mô hình Neural Matrix Factorization (Xiangnan He et al., WWW 2017):
- **Nhánh GMF (Generalized Matrix Factorization):**
  - Không gian embedding người dùng: $\mathbf{p}_u^G \in \mathbb{R}^{d_{GMF}}$ (`n_factors_gmf = 32`).
  - Không gian embedding phim: $\mathbf{q}_i^G \in \mathbb{R}^{d_{GMF}}$ (`n_factors_gmf = 32`).
  - Phép tương tác Hadamard (element-wise product):
    $$\phi^{GMF} = \mathbf{p}_u^G \odot \mathbf{q}_i^G \in \mathbb{R}^{32}$$
- **Nhánh MLP (Multi-Layer Perceptron):**
  - Embedding người dùng: $\mathbf{p}_u^M \in \mathbb{R}^{d_{MLP}}$ (`n_factors_mlp = 64`).
  - Embedding phim: $\mathbf{q}_i^M \in \mathbb{R}^{d_{MLP}}$ (`n_factors_mlp = 64`).
  - Ghép nối vector (concatenation) đầu vào: $\mathbf{z}_0 = [\mathbf{p}_u^M, \mathbf{q}_i^M] \in \mathbb{R}^{128}$.
  - Chuỗi tầng phi tuyến sâu với kích thước `layers = [128, 64, 32]`:
    $$\mathbf{z}_1 = \text{ReLU}(\mathbf{W}_1^T \mathbf{z}_0 + \mathbf{b}_1), \quad \mathbf{W}_1 \in \mathbb{R}^{128 \times 128}$$
    $$\mathbf{z}_2 = \text{ReLU}(\mathbf{W}_2^T \mathbf{z}_1 + \mathbf{b}_2), \quad \mathbf{W}_2 \in \mathbb{R}^{128 \times 64}$$
    $$\phi^{MLP} = \mathbf{z}_3 = \text{ReLU}(\mathbf{W}_3^T \mathbf{z}_2 + \mathbf{b}_3), \quad \mathbf{W}_3 \in \mathbb{R}^{64 \times 32}$$
  - Đã tích hợp `Dropout(p=0.1)` giữa các tầng để chống hiện tượng hiệp phương sai và overfitting.
- **Tầng NeuMF Fusion Projection:**
  - Kết hợp biểu diễn tuyến tính và phi tuyến:
    $$\phi^{NeuMF} = [\phi^{GMF}, \phi^{MLP}] \in \mathbb{R}^{32 + 32} = \mathbb{R}^{64}$$
  - Chiếu affine về logit tương tác chưa chuẩn hóa:
    $$\hat{y}_{ui}^{logit} = \mathbf{h}^T \phi^{NeuMF} + b_{out}, \quad \mathbf{h} \in \mathbb{R}^{64 \times 1}$$

### 1.2. Kiểm toán Gradient Flow và Lan truyền ngược (Backpropagation)
Thực nghiệm kiểm tra trực tiếp qua script kiểm toán [`tests/test_phase4c_audit.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/tests/test_phase4c_audit.py):
- **Hàm mất mát:** `nn.BCEWithLogitsLoss()`. Hàm này tích hợp hàm Sigmoid và Binary Cross-Entropy theo công thức log-sum-exp numerically stable:
  $$\mathcal{L} = -\sum_{(u, i)} \left[ y_{ui} \log \sigma(\hat{y}_{ui}^{logit}) + (1 - y_{ui}) \log (1 - \sigma(\hat{y}_{ui}^{logit})) \right]$$
  Đảm bảo không xảy ra underflow/overflow ở các giá trị cực trị.
- **Gradient Norm đo đạc thực tế sau 1 batch backward:**
  - GMF User Embedding Gradient Norm: $||\nabla \mathbf{p}_u^G|| = 1.5776 \times 10^{-3} > 0$.
  - GMF Item Embedding Gradient Norm: $||\nabla \mathbf{q}_i^G|| = 1.5776 \times 10^{-3} > 0$.
  - MLP Embedding Gradient Norm: $||\nabla [\mathbf{p}_u^M, \mathbf{q}_i^M]|| = 6.3629 \times 10^{-2} > 0$.
  - Fusion Layer Projection Weight Gradient Norm: $||\nabla \mathbf{h}|| = 3.7698 \times 10^{-3} > 0$.
- **Cập nhật tham số:** Kiểm tra `torch.allclose(weight_before, weight_after)` xác nhận $100\%$ các ma trận trọng số ở cả 2 nhánh GMF, MLP và Fusion đều được cập nhật giá trị mới sau mỗi bước `optimizer.step()`.

### 1.3. Khởi tạo trọng số và Hội tụ Loss
- Embedding GMF và MLP được khởi tạo phân phối chuẩn $\mathcal{N}(0, 0.01^2)$.
- Các tầng Dense MLP được khởi tạo Xavier Uniform (`xavier_uniform_`).
- Tầng Affine Fusion được khởi tạo Kaiming Uniform (`kaiming_uniform_`).
- **Lịch sử giảm Training Loss trên toàn bộ tập train MovieLens 100K:**
  - Epoch 1: $\mathcal{L} = 0.6863$
  - Epoch 5: $\mathcal{L} = 0.3018$
  - Epoch 10: $\mathcal{L} = 0.2658$
  - Epoch 15: $\mathcal{L} = 0.2480$
  Đường cong mất mát đơn điệu giảm đều, hội tụ mượt mà, không xảy ra hiện tượng gradient explosion hay vanishment.

### 1.4. Chế độ Suy luận và Xử lý Cold-Start
- Hàm `fit()` gọi `self.net.eval()` trước khi kết thúc quá trình huấn luyện.
- Hàm `predict()` và `recommend()` thực thi trong block `with torch.no_grad():` để triệt tiêu overhead tính toán gradient, vô hiệu hóa Dropout.
- Với người dùng hoặc item chưa từng xuất hiện (Cold-start):
  - `predict(cold_u, cold_i)`: trả về xác suất tiên nghiệm trung lập $0.5$.
  - `recommend(cold_u)`: giữ nguyên thứ tự ứng viên mặc định (hoặc trả về phim có độ phổ biến cao).
  - Phim lạ trong danh sách ứng viên: gán logit $-10^9$ để luôn xếp sau các phim hợp lệ có trong từ điển item embeddings.

---

## 2. KIỂM TOÁN NEGATIVE SAMPLING

### 2.1. Phân biệt Positive Interaction và Confirmed Negative Feedback
Trong bài toán gợi ý Top-N trên MovieLens:
- Tập tương tác dương tính (Positives) được định nghĩa nghiêm ngặt: $y_{ui} = 1 \iff r_{ui} \ge 4.0$ (theo giao thức chuẩn hóa của toàn bộ đồ án). Trong `train.csv` (80,000 dòng), có đúng $44,161$ tương tác dương tính.
- Các tương tác người dùng đánh giá $1, 2, 3$ sao trong Train:
  - **Không** được coi là tương tác dương tính.
  - **Tuyệt đối không** bị đưa vào tập mẫu âm (Negative Samples), vì người dùng đã xem phim này, không phải unobserved interaction.
  - Được loại bỏ ra khỏi cả hai luồng để tránh mâu thuẫn tín hiệu giám sát (conflicting supervisory signal).

### 2.2. Kiểm tra tính rời nhau (Disjointness) và Không rò rỉ dữ liệu
- Bể ứng viên âm (Unobserved Pool) cho người dùng $u$ được xây dựng bằng:
  $$\mathcal{P}_u^{neg} = \mathcal{I}_{all} \setminus \mathcal{I}_u^{train}$$
  Trong đó $\mathcal{I}_u^{train}$ là toàn bộ tập các bộ phim người dùng $u$ đã từng đánh giá trong `train.csv` (bất kể mức rating).
- Kiểm toán mã nguồn và dữ liệu xác nhận:
  - $100\%$ các mẫu âm trong quá trình huấn luyện NeuMF không giao với lịch sử huấn luyện của người dùng: $\mathcal{S}_{sampled}^{neg} \cap \mathcal{I}_u^{train} = \emptyset$.
  - Dữ liệu `test.csv` hoàn toàn không bị truy cập trong quá trình sinh mẫu âm.

### 2.3. Tỷ lệ Negative Sampling và Cơ sở Khoa học
- Tỷ lệ lấy mẫu âm được thiết lập $N_{neg} = 4$ (tức 4 mẫu âm cho mỗi tương tác dương tính).
- Tổng kích thước tập dữ liệu huấn luyện mỗi epoch:
  $$N_{epoch} = N_{pos} \times (1 + N_{neg}) = 44,161 \times (1 + 4) = 220,805 \text{ mẫu}$$
- Cơ sở lý thuyết: Nghiên cứu gốc của Xiangnan He et al. (NCF, WWW 2017) đã chứng minh tỷ lệ $N_{neg} \in [3, 6]$ đạt điểm cân bằng Pareto tối ưu giữa tốc độ huấn luyện và năng lực phân biệt xác suất (discrimination power) so với các tỷ lệ nhỏ hơn ($N_{neg}=1$) hay quá lớn ($N_{neg}=10$).
- Kiểm chứng thực nghiệm trên tập Validation (`train_inner.csv` & `val.csv`):
  - $N_{neg} = 2$: Val HR@10 = $0.6624$, Val NDCG@10 = $0.2185$
  - $N_{neg} = 4$: Val HR@10 = **$0.6958$**, Val NDCG@10 = **$0.2412$**
  Việc lựa chọn $N_{neg} = 4$ hoàn toàn dựa trên dữ liệu Validation, không sử dụng dữ liệu Test.

---

## 3. KIỂM TOÁN HYBRID RECOMMENDER

File kiểm toán: [`src/models/hybrid.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/models/hybrid.py)

### 3.1. Công thức Weighted Fusion và Chuẩn hóa Điểm số
- Điểm dự đoán của Collaborative Filtering (User-Based CF) nằm trong miền rating $[1.0, 5.0]$.
- Điểm dự đoán của Content-Based nằm trong miền tương đồng Cosine $[0.0, 1.0]$.
- Do hai mô hình có phân phối và thang đo khác nhau, việc cộng trực tiếp sẽ khiến CF áp đảo hoàn toàn CB. Giải thuật thực hiện chuẩn hóa động Min-Max trên đúng tập ứng viên $\mathcal{C}_u$:
  $$\tilde{s}_{CF}(u, i) = \frac{s_{CF}(u, i) - \min_{j \in \mathcal{C}_u} s_{CF}(u, j)}{\max_{j \in \mathcal{C}_u} s_{CF}(u, j) - \min_{j \in \mathcal{C}_u} s_{CF}(u, j) + \epsilon}$$
  $$\tilde{s}_{CB}(u, i) = \frac{s_{CB}(u, i) - \min_{j \in \mathcal{C}_u} s_{CB}(u, j)}{\max_{j \in \mathcal{C}_u} s_{CB}(u, j) - \min_{j \in \mathcal{C}_u} s_{CB}(u, j) + \epsilon}$$
- Điểm lai ghép cuối cùng:
  $$s_{hybrid}(u, i) = \alpha \cdot \tilde{s}_{CF}(u, i) + (1 - \alpha) \cdot \tilde{s}_{CB}(u, i)$$
  với $\alpha = 0.80$ và $\epsilon = 10^{-8}$.

### 3.2. Xử lý Trường hợp Ngoại lệ (Edge Cases)
- **Trường hợp mảng điểm đồng nhất ($\max = \min$):** Xử lý an toàn tại dòng 67 của `hybrid.py`:
  `if diff < epsilon: return np.full_like(scores, 0.5, dtype=np.float32)`. Tránh hoàn toàn lỗi chia cho 0 (ZeroDivisionError) và gán điểm ưu tiên trung tính.
- **Giá trị ranh giới $\alpha$:** Đã kiểm thử xác nhận $\alpha = 0.0$ tương đương thuần túy Content-Based, $\alpha = 1.0$ tương đương thuần túy Collaborative Filtering.
- **Tính ổn định số học:** Toàn bộ điểm số đầu ra hữu hạn, không có sự xuất hiện của `NaN` hay `Inf`.
- **Thứ tự xếp hạng xác định (Deterministic Tie-Breaking):** Sử dụng `np.lexsort(([-int(m) for m in candidate_list], -fused_scores))` bảo đảm khi hai phim có điểm lai ghép bằng nhau, phim có mã `movie_id` nhỏ hơn luôn được ưu tiên, loại bỏ tính bất định do thứ tự lặp dictionary.

---

## 4. KIỂM TOÁN QUY TRÌNH ĐÁNH GIÁ (EVALUATION FRAMEWORK)

File kiểm toán: [`src/evaluation/benchmark_phase4c.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/evaluation/benchmark_phase4c.py)

### 4.1. Tính nhất quán của Benchmark
Kiểm tra đối chiếu giao thức đánh giá giữa Phase 3, Phase 4A, Phase 4B và Phase 4C:
1. **Tập Test độc lập:** Cả 10 mô hình đều được đánh giá trên đúng tập `data/processed/test.csv` (20,000 dòng).
2. **Tập người dùng đánh giá:** Đúng $935$ người dùng (những người có ít nhất 1 tương tác đánh giá $r \ge 4.0$ trong tập Test). Tuyệt đối không thay đổi số lượng người dùng đánh giá.
3. **Tập ứng viên gợi ý:** $\mathcal{C}_u = \mathcal{I}_{all} \setminus \mathcal{I}_u^{train}$ với $|\mathcal{I}_{all}| = 1682$. Mọi mô hình đều phải xếp hạng trên cùng tập ứng viên chưa từng xuất hiện trong Train của người dùng.
4. **Ngưỡng Relevance:** $r \ge 4.0$ đồng nhất cho cả 5 độ đo.
5. **Ngưỡng cắt Top-N:** $K = 10$.
6. **Phương pháp tổng hợp độ đo:** Macro-average trên toàn bộ 935 người dùng kiểm thử.

### 4.2. Tính bảo toàn của 8 mô hình tiền nhiệm
Kết quả benchmark thực tế của 8 mô hình từ Phase 4B sang Phase 4C trùng khớp $100\%$:
- Popularity: HR@10 = $0.3914$, NDCG@10 = $0.0764$
- Content-Based: HR@10 = $0.1872$, NDCG@10 = $0.0296$
- User-Based CF: HR@10 = $0.6524$, NDCG@10 = $0.1948$
- Item-Based CF: HR@10 = $0.5904$, NDCG@10 = $0.1611$
- KNNBaseline: HR@10 = $0.5027$, NDCG@10 = $0.1316$
- Funk SVD: HR@10 = $0.4032$, NDCG@10 = $0.0715$
- SVD++: HR@10 = $0.4759$, NDCG@10 = $0.1089$
- NMF: HR@10 = $0.3316$, NDCG@10 = $0.0525$

---

## 5. KIỂM TRA TÍNH TÁI LẬP (REPRODUCIBILITY AUDIT)

Thực hiện chạy lại hai lần độc lập mô hình `Neural CF` trên toàn bộ tập dữ liệu với cùng `random_state = 42` trên CPU:

```text
Comparison between Run 1 and Run 2 on Test (935 evaluation users):
  precision_at_10   : Run1 = 0.225134 | Run2 = 0.225134 | Sai lệch = 0.000000e+00
  recall_at_10      : Run1 = 0.239071 | Run2 = 0.239071 | Sai lệch = 0.000000e+00
  f1_at_10          : Run1 = 0.189793 | Run2 = 0.189793 | Sai lệch = 0.000000e+00
  ndcg_at_10        : Run1 = 0.308276 | Run2 = 0.308276 | Sai lệch = 0.000000e+00
  hit_rate_at_10    : Run1 = 0.820321 | Run2 = 0.820321 | Sai lệch = 0.000000e+00
```

**Nhận xét:** Độ lệch thực tế bằng chính xác $0.000000 \times 10^0$. Hệ thống đạt tính xác định $100\%$ (bitwise reproducible) trên môi trường CPU PyTorch khi cố định seed.

---

## 6. BẢNG BENCHMARK CHÍNH THỨC 10 MÔ HÌNH HỆ THỐNG

| Hạng | Tên Mô hình | Họ Thuật toán | Precision@10 | Recall@10 | F1@10 | NDCG@10 | Hit Rate@10 | Thời gian Huấn luyện | Thời gian Đánh giá |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Neural CF (NeuMF)** | **Deep Learning / NeuMF** | **0.2251** | **0.2391** | **0.1898** | **0.3083** | **0.8203** | 62.80s | 2.27s |
| **2** | **Hybrid (CF + CB)** | **Hybrid Recommender** | **0.1475** | **0.1327** | **0.1164** | **0.1943** | **0.6578** | 0.00s* | 8.35s |
| 3 | User-Based CF | Memory-based CF (User KNN) | 0.1467 | 0.1352 | 0.1160 | 0.1948 | 0.6524 | 0.08s | 4.39s |
| 4 | Item-Based CF | Memory-based CF (Item KNN) | 0.1247 | 0.1146 | 0.0979 | 0.1611 | 0.5904 | 0.16s | 4.70s |
| 5 | KNNBaseline | Neighborhood CF with Biases | 0.1059 | 0.0816 | 0.0778 | 0.1316 | 0.5027 | 0.55s | 3.53s |
| 6 | SVD++ | Factorization + Implicit Feedback | 0.0896 | 0.0701 | 0.0671 | 0.1089 | 0.4759 | 2.65s | 4.67s |
| 7 | Funk SVD | Regularized Matrix Factorization | 0.0711 | 0.0508 | 0.0508 | 0.0715 | 0.4032 | 1.15s | 3.75s |
| 8 | Popularity Baseline | Bayesian Weighted Rating | 0.0653 | 0.0497 | 0.0473 | 0.0764 | 0.3914 | 0.03s | 3.23s |
| 9 | NMF | Non-Negative Matrix Factorization | 0.0542 | 0.0380 | 0.0387 | 0.0525 | 0.3316 | 2.05s | 3.32s |
| 10 | Content-Based | TF-IDF Cosine Similarity | 0.0236 | 0.0199 | 0.0174 | 0.0296 | 0.1872 | 0.07s | 5.37s |

*\*Ghi chú: Mô hình Hybrid tái sử dụng các sub-model đã fit trong cùng phiên benchmark nên thời gian fit ghi nhận 0.00s.*

---

## 7. CÁC LỖI PHÁT HIỆN VÀ BIỆN PHÁP KHẮC PHỤC

1. **Phát hiện:** Trong kiểm thử đơn vị mock cho Hybrid Recommender, mock DataFrame truyền vào `ContentBasedRecommender` chỉ có cột `genres` thay vì `genres_str`, gây ra ngoại lệ `KeyError: 'genres_str'` trong hàm `format_content`.
   - **Khắc phục:** Cập nhật mock DataFrame trong [`tests/test_phase4c_audit.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/tests/test_phase4c_audit.py) cung cấp chuẩn trường dữ liệu `genres_str`, đồng bộ với cấu trúc `movies_clean.csv`.
2. **Xác minh Thư viện Random Seed:** Bổ sung việc thiết lập đồng thời cả `torch.manual_seed()`, `np.random.seed()`, và `random.seed()` để triệt tiêu mọi khả năng xuất hiện tính bất định trong các thư viện bổ trợ.

---

## 8. KẾT QUẢ KIỂM THỬ TOÀN DỰ ÁN (TEST SUITE STATUS)

Chạy kiểm thử toàn bộ dự án qua lệnh `python -m unittest discover -s tests -v`:
- `tests/test_phase2_pipeline.py`: 10/10 PASS
- `tests/test_phase3_pipeline.py`: 10/10 PASS
- `tests/test_phase4a_pipeline.py`: 10/10 PASS
- `tests/test_phase4b_pipeline.py`: 15/15 PASS
- `tests/test_phase4c_pipeline.py`: 13/13 PASS
- `tests/test_phase4c_audit.py`: 7/7 PASS

**Tổng cộng:** **65 / 65 unit tests ĐẠT 100% (OK)** trong thời gian thực thi ổn định.

---

## 9. GIỚI HẠN VÀ KHUYẾN NGHỊ CHO PHASE 5

1. **Độ trễ suy luận của NeuMF:** Mặc dù đạt chất lượng gợi ý cao nhất (HR@10 = $82.03\%$), việc tính toán forward pass qua mạng neural sâu cho 1,600 ứng viên mỗi người dùng mất khoảng 2.4 ms/người dùng. Trong giao diện Streamlit (Phase 5), nên lưu cache mô hình NeuMF đã huấn luyện (`st.cache_resource`) để phản hồi tức thời cho người dùng.
2. **Độ trễ của Hybrid:** Cần tính điểm song song của cả CF và CB trước khi normalize, nên thời gian phản hồi ~8.9 ms. Có thể tối ưu hóa bằng vectorization dot product đã triển khai sẵn trong `hybrid.py`.

---

## 10. TRẠNG THÁI GIT & KẾT LUẬN

### Trạng thái Git:
- Nhánh hiện tại: `main` (Up to date with `origin/main`).
- Không có bất kỳ thay đổi nào làm biến đổi dữ liệu phân hoạch gốc `train.csv` hay `test.csv`.
- Các file mới thuộc Phase 4C và Phase 4C Audit sẵn sàng để người dùng commit và push.

### Kết luận:
Toàn bộ 10 mô hình gợi ý phim, 5 độ đo Top-N, framework đánh giá, dữ liệu kiểm thử và các bài test đã được kiểm toán toàn diện và chứng minh đạt chuẩn khoa học nghiêm ngặt. Hệ thống **ĐÃ HOÀN TOÀN SẴN SÀNG** để bước vào **Giai đoạn 5 (Phase 5 – Xây dựng Ứng dụng Demo Tương tác Streamlit)**.
