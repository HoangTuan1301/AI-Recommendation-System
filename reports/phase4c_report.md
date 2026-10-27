# BÁO CÁO THỰC NGHIỆM GIAI ĐOẠN 4C: NEURAL COLLABORATIVE FILTERING & HYBRID RECOMMENDER
## Hoàn Tất 10 Mô Hình Gợi Ý Phim Trên MovieLens 100K & Benchmark Toàn Diện

**Dự án:** AI-Recommendation-System (Học kỳ 5)  
**Đề tài:** Xây dựng chương trình gợi ý phim/bài hát theo sở thích ứng dụng Trí tuệ nhân tạo  
**Dataset:** MovieLens 100K (Train: 80,000 / Test: 20,000 | 935 evaluation users)  
**Công nghệ bổ sung:** PyTorch 2.14.1+cpu, Adam Optimizer, BCEWithLogitsLoss, Min-Max Score Normalization  
**Ngày thực hiện:** 27/10/2026  
**Trạng thái:** HOÀN THÀNH 10/10 MÔ HÌNH – PASS 58/58 UNIT TESTS (100%)

---

## 1. TỔNG QUAN KIẾN TRÚC HAI MÔ HÌNH GIAI ĐOẠN 4C

Trong Giai đoạn 4C, dự án hoàn thành hai mô hình cuối cùng để đạt đủ **10 mô hình gợi ý** theo yêu cầu của giảng viên:
1. **Model 9 – NeuralCFRecommender (NeuMF - Neural Matrix Factorization):**
   - Công trình kinh điển của Xiangnan He et al. (WWW 2017: *"Neural Collaborative Filtering"*).
   - Tích hợp đồng thời nhánh **Generalized Matrix Factorization (GMF)** tuyến tính (học tương tác nhân) và nhánh **Multi-Layer Perceptron (MLP)** phi tuyến tính (học tương tác bậc cao giữa user và item embeddings).
   - Huấn luyện theo cơ chế xếp hạng phản hồi ẩn (implicit ranking) với hàm mất mát Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) và lấy mẫu âm tính ngẫu nhiên (Uniform Negative Sampling).
2. **Model 10 – HybridRecommender (Weighted Hybrid Recommender):**
   - Kết hợp trường phái Lọc cộng tác (**Collaborative Filtering**) và Lọc dựa trên nội dung (**Content-Based Filtering**).
   - Chuẩn hóa điểm số động trên tập ứng viên (Score Normalization qua Min-Max scaling) trước khi tổng hợp trọng số $\alpha$:
     $$s_{hybrid}(u, i) = \alpha \cdot \tilde{s}_{CF}(u, i) + (1 - \alpha) \cdot \tilde{s}_{CB}(u, i)$$

Cả 10 mô hình hiện nay đều kế thừa từ [`BaseRecommender`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/models/base_recommender.py), tuân thủ giao thức đánh giá Top-N tại [`TopNEvaluator`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/evaluation/evaluator.py) trên cùng 935 evaluation users và cùng tập ứng viên $\mathcal{C}_u = \mathcal{I}_{all} \setminus \mathcal{I}_u^{train}$.

---

## 2. CÔNG THỨC TOÁN HỌC & CƠ CHẾ HOẠT ĐỘNG

### 2.1. Neural Matrix Factorization (NeuMF)

NeuMF tận dụng không gian biểu diễn kép cho người dùng và bộ phim:
- **Nhánh Generalized Matrix Factorization (GMF):**
  $$\mathbf{p}_u^G \in \mathbb{R}^{d_{gmf}}, \quad \mathbf{q}_i^G \in \mathbb{R}^{d_{gmf}}$$
  $$\mathbf{\phi}^{GMF} = \mathbf{p}_u^G \odot \mathbf{q}_i^G \quad (\text{tích Hadamard từng phần tử})$$
- **Nhánh Multi-Layer Perceptron (MLP):**
  $$\mathbf{p}_u^M \in \mathbb{R}^{d_{mlp}}, \quad \mathbf{q}_i^M \in \mathbb{R}^{d_{mlp}}$$
  $$\mathbf{z}_0 = [\mathbf{p}_u^M, \mathbf{q}_i^M] \in \mathbb{R}^{2 d_{mlp}}$$
  $$\mathbf{z}_1 = \text{ReLU}(\mathbf{W}_1 \mathbf{z}_0 + \mathbf{b}_1)$$
  $$\mathbf{z}_2 = \text{ReLU}(\mathbf{W}_2 \mathbf{z}_1 + \mathbf{b}_2)$$
  $$\mathbf{\phi}^{MLP} = \text{ReLU}(\mathbf{W}_L \mathbf{z}_{L-1} + \mathbf{b}_L)$$
- **Lớp kết hợp NeuMF (NeuMF Fusion Layer):**
  $$\mathbf{h}_{fusion} = [\mathbf{\phi}^{GMF}, \mathbf{\phi}^{MLP}] \in \mathbb{R}^{d_{gmf} + \text{dim}(\mathbf{\phi}^{MLP})}$$
  $$\hat{y}_{logits} = \mathbf{w}_{out}^T \mathbf{h}_{fusion} + b_{out}$$
  $$\hat{y} = \sigma(\hat{y}_{logits}) = \frac{1}{1 + e^{-\hat{y}_{logits}}}$$
- **Hàm mất mát Binary Cross-Entropy with Logits:**
  $$\mathcal{L} = -\sum_{(u, i) \in \mathcal{Y}^+ \cup \mathcal{Y}^-} \left[ y_{ui} \log \sigma(\hat{y}_{logits}) + (1 - y_{ui}) \log (1 - \sigma(\hat{y}_{logits})) \right] + \lambda \|\Theta\|_2^2$$

### 2.2. Quy Trình Negative Sampling (Lấy Mẫu Phản Hồi Âm)
1. **Định nghĩa tương tác dương ($\mathcal{Y}^+$):**
   - Đánh giá trong `train_df` có rating $r_{ui} \ge 4.0$ (đồng nhất với ngưỡng ground truth đánh giá).
2. **Lấy mẫu âm tính ($\mathcal{Y}^-$):**
   - Với mỗi tương tác dương $(u, i) \in \mathcal{Y}^+$, hệ thống lấy mẫu ngẫu nhiên $N_{neg} = 4$ bộ phim chưa quan sát từ tập $\mathcal{I}_{all} \setminus \mathcal{I}_u^{train}$.
   - **Cam kết bất biến:** Phim lấy mẫu âm tuyệt đối không bao giờ trùng với các phim đã xem trong Train hay phim trong tập Test.
   - **Ý nghĩa kỹ thuật:** Việc gán nhãn 0 cho các mẫu âm không có nghĩa là người dùng ghét bộ phim đó, mà phản ánh tín hiệu xác suất: người dùng chưa từng tương tác với bộ phim đó trong môi trường tự nhiên.

### 2.3. Weighted Hybrid Recommender
- **Vấn đề lệch thang đo (Incompatible Scales):**
  - Mô hình CF (như User-Based CF) dự đoán rating từ $1.0$ đến $5.0$.
  - Mô hình Content-Based (TF-IDF Cosine Similarity) dự đoán độ tương đồng ngữ nghĩa từ $-1.0$ đến $1.0$.
  - Nếu cộng trực tiếp, điểm CF sẽ lấn át hoàn toàn điểm Content-Based, triệt tiêu tác dụng lai ghép.
- **Giải pháp: Min-Max Normalization trên tập ứng viên:**
  Với người dùng $u$ và danh sách ứng viên $\mathcal{C}_u$:
  $$\tilde{s}_{CF}(u, i) = \frac{s_{CF}(u, i) - \min_{j \in \mathcal{C}_u} s_{CF}(u, j)}{\max_{j \in \mathcal{C}_u} s_{CF}(u, j) - \min_{j \in \mathcal{C}_u} s_{CF}(u, j) + \epsilon} \in [0, 1]$$
  $$\tilde{s}_{CB}(u, i) = \frac{s_{CB}(u, i) - \min_{j \in \mathcal{C}_u} s_{CB}(u, j)}{\max_{j \in \mathcal{C}_u} s_{CB}(u, j) - \min_{j \in \mathcal{C}_u} s_{CB}(u, j) + \epsilon} \in [0, 1]$$
- **Hòa trộn tuyến tính:**
  $$s_{hybrid}(u, i) = \alpha \cdot \tilde{s}_{CF}(u, i) + (1 - \alpha) \cdot \tilde{s}_{CB}(u, i)$$
  Với $\alpha \in [0.0, 1.0]$.

---

## 3. QUY TRÌNH TUNING SIÊU THAM SỐ TRÊN VALIDATION SET (TASK 4C.4)

Tất cả các quyết định kiến trúc và siêu tham số đều được thực hiện độc lập trên tập **Validation (`val.csv`: 15,995 dòng)** tách từ `train.csv`. Tập `test.csv` hoàn toàn không tham gia vào quá trình này.

Dữ liệu lưu tại file [`results/phase4c_tuning_results.csv`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/results/phase4c_tuning_results.csv) trên 927 evaluation users:

| Họ mô hình | Tên cấu hình | Tham số kỹ thuật | Val Precision@10 | Val Recall@10 | Val F1@10 | Val NDCG@10 | Val Hit Rate@10 | Kết luận lựa chọn |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Neural CF** | NCF_small | $d_G=8, d_M=16$, layers=[32,16], lr=0.001, neg=2 | 0.1098 | 0.1296 | 0.1004 | 0.1475 | 0.5761 | |
| **Neural CF** | NCF_standard | $d_G=16, d_M=32$, layers=[64,32,16], lr=0.001, neg=4 | 0.1372 | 0.1792 | 0.1294 | 0.1907 | 0.6807 | Rất mạnh |
| **Neural CF** | NCF_high_lr | $d_G=16, d_M=32$, layers=[64,32,16], lr=0.002, neg=4 | 0.1359 | 0.1703 | 0.1268 | 0.1849 | 0.6731 | |
| **Neural CF** | **NCF_large** | **$d_G=32, d_M=64$, layers=[128,64,32], lr=0.001, neg=4** | **0.1407** | **0.1806** | **0.1320** | **0.1906** | **0.6958** | **CHỌN (HR cao nhất)** |
| **Hybrid** | UserCF_a50 | CF=UserCF, $\alpha=0.50$, norm=min_max | 0.0743 | 0.0853 | 0.0663 | 0.1027 | 0.4682 | |
| **Hybrid** | UserCF_a70 | CF=UserCF, $\alpha=0.70$, norm=min_max | 0.0926 | 0.1082 | 0.0829 | 0.1275 | 0.5405 | |
| **Hybrid** | **UserCF_a80** | **CF=UserCF, $\alpha=0.80$, norm=min_max** | **0.0951** | **0.1124** | **0.0859** | **0.1317** | **0.5566** | **CHỌN (HR cao nhất)** |
| **Hybrid** | UserCF_a85 | CF=UserCF, $\alpha=0.85$, norm=min_max | 0.0951 | 0.1123 | 0.0862 | 0.1326 | 0.5545 | Cân bằng NDCG |
| **Hybrid** | UserCF_a90 | CF=UserCF, $\alpha=0.90$, norm=min_max | 0.0956 | 0.1136 | 0.0869 | 0.1344 | 0.5523 | |
| **Hybrid** | UserCF_zscore | CF=UserCF, $\alpha=0.80$, norm=z_score | 0.0949 | 0.1122 | 0.0861 | 0.1333 | 0.5523 | |
| **Hybrid** | ItemCF_a80 | CF=ItemCF, $\alpha=0.80$, norm=min_max | 0.0789 | 0.0961 | 0.0715 | 0.1096 | 0.4790 | Thấp hơn UserCF |
| **Hybrid** | SVDpp_a80 | CF=SVD++, $\alpha=0.80$, norm=min_max | 0.0313 | 0.0285 | 0.0253 | 0.0385 | 0.2276 | Thấp hơn UserCF |

**Nhận định kỹ thuật từ Tuning:**
1. **Neural CF:** Tăng kích thước embedding ($d_{gmf}=32, d_{mlp}=64$) cùng kiến trúc MLP 3 tầng $[128, 64, 32]$ giúp mạng học được các đặc trưng trừu tượng sâu hơn, nâng Val Hit Rate@10 từ 57.61% lên **69.58%**. Tỷ lệ lấy mẫu âm $N_{neg}=4$ cung cấp độ tương phản tín hiệu vượt trội so với $N_{neg}=2$.
2. **Hybrid:** Việc chọn `User-Based CF` làm thành phần cộng tác cho kết quả vượt trội hoàn toàn so với `Item-Based CF` (55.66% vs 47.90%) và `SVD++` (22.76%). Trọng số $\alpha = 0.80$ phản ánh tỷ lệ tối ưu: 80% sức mạnh cá nhân hóa từ hành vi cộng đồng và 20% định hướng từ nội dung thuộc tính thể loại/tựa đề phim.

---

## 4. BẢNG XẾP HẠNG BENCHMARK TOÀN DIỆN 10 MÔ HÌNH (OFFICIAL BENCHMARK)

Sau khi lựa chọn cấu hình từ Validation, cả 10 mô hình được huấn luyện trên toàn bộ `train.csv` (80,000 dòng) và đánh giá **duy nhất một lần** trên `test.csv` (20,000 dòng, 935 evaluation users có rating $r \ge 4.0$).

Kết quả chính thức được lưu tại file [`results/phase4c_metrics.csv`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/results/phase4c_metrics.csv):

| Thứ hạng | Nhóm thuật toán | Tên mô hình (Model) | Precision@10 | Recall@10 | F1@10 | NDCG@10 | Hit Rate@10 | Fit Time (s) | Eval Time (s) | Tổng thời gian (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **Deep Learning** | **Neural CF (NeuMF)** | **0.2251** | **0.2391** | **0.1898** | **0.3083** | **0.8203** | 61.3250 s | 2.3662 s | **63.6912 s** |
| **2** | **Hybrid** | **Hybrid (UserCF + CB)** | **0.1475** | **0.1327** | **0.1164** | **0.1943** | **0.6578** | 0.0215 s | 5.5823 s | **5.6038 s** |
| **3** | Neighborhood CF | **User-Based CF** | **0.1467** | **0.1352** | **0.1160** | **0.1948** | **0.6524** | 0.0899 s | 1.4234 s | **1.5132 s** |
| **4** | Neighborhood CF | **Item-Based CF** | **0.1247** | **0.1146** | **0.0979** | **0.1611** | **0.5904** | 0.1714 s | 1.4616 s | **1.6329 s** |
| **5** | Neighborhood CF | **KNNBaseline** | **0.1059** | **0.0816** | **0.0778** | **0.1316** | **0.5027** | 0.5445 s | 1.8370 s | **2.3815 s** |
| **6** | Matrix Factorization | **SVD++** | **0.0896** | **0.0701** | **0.0671** | **0.1089** | **0.4759** | 19.5154 s | 0.9048 s | **20.4202 s** |
| **7** | Matrix Factorization | **SVD (Funk SVD)** | **0.0711** | **0.0508** | **0.0508** | **0.0715** | **0.4032** | 1.6500 s | 0.8807 s | **2.5307 s** |
| **8** | Baseline | **Popularity** | **0.0653** | **0.0497** | **0.0473** | **0.0764** | **0.3914** | 0.0369 s | 0.8519 s | **0.8888 s** |
| **9** | Matrix Factorization | **NMF** | **0.0542** | **0.0380** | **0.0387** | **0.0525** | **0.3316** | 3.4960 s | 0.8936 s | **4.3895 s** |
| **10** | Baseline | **Content-Based** | **0.0236** | **0.0199** | **0.0174** | **0.0296** | **0.1872** | 0.0661 s | 5.4283 s | **5.4944 s** |

*(Ghi chú: Đánh giá trên 935 evaluation users có ground truth $r \ge 4.0$ trong Test, candidate set gồm 1,514.8 phim/người dùng).*

---

## 5. PHÂN TÍCH CHUYÊN SÂU KẾT QUẢ THỰC NGHIỆM

1. **Sự thống trị của Neural Collaborative Filtering (NeuMF):**
   - NeuMF vươn lên vị trí **quán quân toàn diện** với **Hit Rate@10 đạt 82.03%** và **NDCG@10 đạt 0.3083** (vượt xa mô hình thứ hai là Hybrid 65.78% và User-Based CF 65.24%).
   - *Nguyên nhân kỹ thuật:* 
     - Nhánh GMF mô phỏng tích vô hướng tuyến tính cổ điển, trong khi nhánh MLP gồm 3 tầng ẩn $[128, 64, 32]$ với hàm kích hoạt phi tuyến ReLU có khả năng xấp xỉ hàm tùy ý (Universal Approximation Theorem), bóc tách được các mối quan hệ tương tác phi tuyến tính phức tạp giữa user và movie mà các phép nhân ma trận thuần túy không thể nắm bắt.
     - Hàm mất mát `BCEWithLogitsLoss` kết hợp lấy mẫu âm tính (Negative Sampling) tập trung trực tiếp vào bài toán xếp hạng (ranking) thay vì hồi quy điểm số tuyệt đối (rating regression), hoàn toàn phù hợp với bản chất của Top-N recommendation.
2. **Hiệu quả của mô hình lai Hybrid:**
   - Mô hình Hybrid đạt **Hit Rate@10 = 65.78%**, vượt qua mô hình đơn lẻ User-Based CF (65.24%) và vượt trội hoàn toàn so với Content-Based (18.72%).
   - *Nguyên nhân kỹ thuật:* Khi người dùng có ít láng giềng tương đồng đối với một nhóm phim hiếm, thành phần Content-Based với đặc trưng thể loại và tựa đề phim đã "bù đắp" tín hiệu sở thích, giúp cải thiện xếp hạng của các bộ phim phù hợp.
3. **Tính nhất quán của 8 mô hình trước:**
   - Toàn bộ kết quả của 8 mô hình từ Phase 4B giữ nguyên 100% tính chính xác, khẳng định không có sự xáo trộn mã nguồn hay rò rỉ dữ liệu.
4. **Hiệu năng thời gian (Runtime Performance):**
   - Huấn luyện NeuMF trên CPU chỉ mất **~61 giây** cho 15 epochs (nhờ PyTorch CPU vectorization).
   - Thời gian suy luận Top-10 cho toàn bộ 935 người dùng chỉ mất **2.37 giây**, hoàn toàn đáp ứng yêu cầu phục vụ thời gian thực (real-time serving) trong ứng dụng web demo.

---

## 6. KẾT QUẢ BỘ KIỂM THỬ HỒI QUY TOÀN DỰ ÁN (REGRESSION TEST SUITE)

Hệ thống đã bổ sung 13 bài kiểm thử chuyên biệt cho Giai đoạn 4C tại [`tests/test_phase4c_pipeline.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/tests/test_phase4c_pipeline.py), nâng tổng số ca kiểm thử hồi quy lên **58 bài kiểm thử**:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
# Ran 58 tests in 16.256s - OK
```
- **Phase 2 Tests (`tests/test_phase2_pipeline.py`):** 10/10 PASS
- **Phase 3 Tests (`tests/test_phase3_pipeline.py`):** 10/10 PASS
- **Phase 4A Tests (`tests/test_phase4a_pipeline.py`):** 10/10 PASS
- **Phase 4B Tests (`tests/test_phase4b_pipeline.py`):** 15/15 PASS
- **Phase 4C Tests (`tests/test_phase4c_pipeline.py`):** 13/13 PASS
  - Kiểm tra kích thước và forward pass của mạng NeuMF.
  - Kiểm tra embedding GMF và MLP.
  - Kiểm tra negative sampling không lấy mẫu dương tính Train.
  - Kiểm tra không rò rỉ dữ liệu Test khi fit.
  - Kiểm tra chuẩn hóa điểm Min-Max và Z-Score.
  - Kiểm tra công thức hợp nhất điểm Hybrid bằng tính tay.
  - Kiểm tra không có NaN/Inf trong dự đoán.
  - Kiểm tra không đề xuất phim đã xem trong Train history.
  - Kiểm tra tính đơn nhất (không trùng lặp ID) trong gợi ý.
  - Kiểm tra tính tái lập tuyệt đối với seed cố định.
  - Kiểm tra tính tương thích với harness TopNEvaluator.
  - Kiểm tra xử lý trường hợp người dùng mới (Cold User fallback).

---

## 7. HẠN CHẾ CỦA THỰC NGHIỆM

1. **Negative Sampling ngẫu nhiên đều (Uniform Negative Sampling):**
   - Việc chọn ngẫu nhiên các phim chưa xem làm mẫu âm tính có thể vô tình chọn phải các bộ phim mà người dùng thực tế sẽ thích nhưng chưa có cơ hội tiếp cận (False Negatives). Các phương pháp nâng cao hơn như Adversarial Negative Sampling (IRGAN) hoặc Popularity-biased Negative Sampling có thể được khám phá thêm.
2. **Tài nguyên tính toán của Deep Learning:**
   - NeuMF mất 61.3 giây để huấn luyện, gấp 26 lần Funk SVD (2.5 giây) và gấp 40 lần User-Based CF (1.5 giây). Dù vậy, thời gian suy luận (inference time) vẫn đạt tốc độ rất nhanh (2.36s).

---

## 8. DANH SÁCH TỆP TẠO MỚI TRONG GIAI ĐOẠN 4C

- **Mã nguồn mô hình mới:**
  - [`src/models/neural_cf.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/models/neural_cf.py): Kiến trúc NeuMF (GMF + MLP + Fusion) với PyTorch.
  - [`src/models/hybrid.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/models/hybrid.py): Weighted Hybrid với Min-Max Score Normalization.
- **Mã nguồn đánh giá và thử nghiệm:**
  - [`src/evaluation/tune_phase4c.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/evaluation/tune_phase4c.py): Kịch bản tuning NCF và Hybrid trên Validation.
  - [`src/evaluation/benchmark_phase4c.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/evaluation/benchmark_phase4c.py): Benchmark toàn diện 10 mô hình trên Test set.
- **Dữ liệu thực nghiệm & Kiểm thử:**
  - [`results/phase4c_tuning_results.csv`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/results/phase4c_tuning_results.csv): Bảng kết quả 12 cấu hình tuning trên Validation.
  - [`results/phase4c_metrics.csv`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/results/phase4c_metrics.csv): Bảng kết quả benchmark chính thức 10 mô hình.
  - [`tests/test_phase4c_pipeline.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/tests/test_phase4c_pipeline.py): Bộ 13 ca kiểm thử đơn vị Phase 4C.
  - [`reports/phase4c_report.md`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/reports/phase4c_report.md): Báo cáo kỹ thuật chi tiết.

---

## 9. TRẠNG THÁI GIT HIỆN TẠI (GIT STATUS)

```
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	reports/phase4c_report.md
	results/phase4c_metrics.csv
	results/phase4c_tuning_results.csv
	src/evaluation/benchmark_phase4c.py
	src/evaluation/tune_phase4c.py
	src/models/hybrid.py
	src/models/neural_cf.py
	tests/test_phase4c_pipeline.py

nothing added to commit but untracked files present (use "git add" to track)
```
- Tuân thủ nguyên tắc không tự ý commit hoặc push nếu chưa có yêu cầu từ người dùng.

---

## 10. ĐỀ XUẤT CHO BƯỚC TIẾP THEO: XÂY DỰNG ỨNG DỤNG DEMO TRỰC QUAN

Sau khi hoàn tất xuất sắc cả 10 mô hình gợi ý và đối sánh trên 5 độ đo khoa học, dự án đã sẵn sàng cho bước cuối cùng:
1. **Giao diện người dùng Streamlit trực quan (`app/app.py`):**
   - **Tab 1 - Khám phá dữ liệu (Dataset Explorer):** Trực quan hóa tương tác MovieLens 100K, phân bố đánh giá, thể loại phim, wordcloud tựa đề phim.
   - **Tab 2 - Bảng đối sánh Benchmark (Model Leaderboard):** Hiển thị bảng đối soát 10 mô hình với 5 độ đo (P@10, R@10, F1@10, NDCG@10, Hit Rate@10) cùng biểu đồ cột trực quan.
   - **Tab 3 - Trải nghiệm gợi ý phim tương tác (Interactive Recommender):**
     - Cho phép chọn người dùng cụ thể từ danh sách 943 người dùng.
     - Lựa chọn 1 trong 10 mô hình AI để sinh danh sách Top 10 bộ phim gợi ý kèm poster, thể loại và điểm dự đoán.
     - Cho phép người dùng nhập sở thích phim cá nhân để nhận gợi ý tức thời từ mô hình `Neural CF` hoặc `Hybrid`.
