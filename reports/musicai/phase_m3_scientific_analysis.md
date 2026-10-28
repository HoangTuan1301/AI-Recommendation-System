# PHÂN TÍCH KHOA HỌC CHUYÊN SÂU — MUSICAI PHASE M3
*(Deep Scientific Analysis: Sparsity, Popularity Bias, User Stratification & Deep Learning vs. Heuristics)*

**Dự án:** Multi-Domain AI Recommendation System (CineAI V2.0 & MusicAI)  
**Tài liệu:** `reports/musicai/phase_m3_scientific_analysis.md`  
**Nhánh Git:** `feature/musicai-models`  
**Ngày thực hiện:** 28/10/2026  
**Chuyên gia thực hiện:** Senior Recommender Systems Researcher & ML Quality Auditor  

---

## 1. SO SÁNH GIỮA THUẬT TOÁN KINH ĐIỂN VÀ MÔ HÌNH HỌC SÂU (TRADITIONAL VS. DEEP LEARNING)

Một trong những kết quả nổi bật nhất của nghiên cứu thực nghiệm Phase M3 là sự vượt trội áp đảo của các phương pháp láng giềng cộng tác truyền thống (**Item-Based CF** và **User-Based CF**) so với mô hình học sâu **Implicit Neural Collaborative Filtering (NeuMF)** trên cả 5 độ đo xếp hạng.

```
HR@10:
  Item-Based CF : ████████████████████████ (0.2448)
  User-Based CF : ███████████████████████  (0.2389)
  Implicit ALS  : █████████████████████    (0.2111)
  BPR-MF        : ████████████████         (0.1643)
  NeuMF (Deep)  : █████████████            (0.1345)
  Popularity    : ██████████               (0.1000)
```

### Nguyên nhân khoa học:
1. **Bản chất của tương tác người dùng - bài hát (Song-to-Song Transition Co-occurrence):**
   - Trong lĩnh vực âm nhạc, hành vi nghe nhạc của người dùng thường mang tính cục bộ cao (người nghe bài này thường chuyển tiếp sang bài kia theo phong cách, tâm trạng hoặc nghệ sĩ). Ma trận tương đồng Cosine của Item-CF nắm bắt trực tiếp và chính xác mối quan hệ đồng nghe (co-listening) này mà không bị suy giảm thông tin qua các lớp nén chiều (dimensionality reduction).
2. **Hiện tượng "Bottleneck" của không gian nhúng (Embedding Bottleneck in High Sparsity):**
   - NeuMF chiếu các ID người dùng và bài hát vào không gian nhúng số chiều thấp ($d = 32$) rồi kết hợp qua MLP phi tuyến. Với ma trận tương tác có độ thưa $99.31\%$, số lượng mẫu dương cho mỗi bài hát là tương đối hạn chế (trung bình ~17 lượt nghe/bài). Điều này khiến mạng nơ-ron khó tối ưu các trọng số của nhiều tầng ẩn (multi-layer perceptron) mà không bị kẹt ở điểm cực tiểu cục bộ hoặc rơi vào xu hướng thiên vị các bài hát phổ biến.
3. **Phù hợp với các nghiên cứu uy tín trong ngành Recommender Systems:**
   - Kết quả này hoàn toàn đồng nhất với các bài báo khoa học nổi tiếng như Dacrema et al. (ACM RecSys 2019 - *"Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches"*): Trên dữ liệu tương tác thưa, các thuật toán Collaborative Filtering truyền thống được điều chỉnh cẩn thận thường xuyên đạt hiệu năng ngang bằng hoặc vượt trội các kiến trúc Deep Learning phức tạp.

---

## 2. ẢNH HƯỞNG CỦA ĐỘ THƯA DỮ LIỆU (DATA SPARSITY IMPACT)

### Thông số độ thưa của bộ dữ liệu MusicAI:
- Số người dùng: $2,520$
- Số bài hát: $2,706$
- Tổng số ô tương tác lý thuyết: $2,520 \times 2,706 = 6,819,120$ ô.
- Số tương tác thực tế trong Train: $36,994$ tương tác.
- **Mật độ ma trận (Matrix Density):**
  $$\text{Density} = \frac{36,994}{6,819,120} \approx 0.005425 = 0.5425\%$$
- **Độ thưa (Matrix Sparsity):**
  $$\text{Sparsity} = 1 - 0.005425 = 99.4575\% \approx 99.46\%$$

### Tác động của độ thưa tới từng nhóm thuật toán:
1. **Thuật toán phân rã ma trận (Matrix Factorization - iALS & BPR-MF):**
   - Phân rã ma trận dựa vào giả định rằng sở thích người dùng có thể biểu diễn qua tổ hợp tuyến tính của các thuộc tính tiềm ẩn (latent factors). Với độ thưa $99.46\%$, iALS vẫn đạt kết quả rất tốt (HR@10 = 0.2111, NDCG@10 = 0.0856) nhờ sử dụng toàn bộ các ô không quan sát làm mẫu âm bản có trọng số thấp ($c_{ui} = 1$) và cập nhật giải tích qua phân rã Cholesky.
2. **Thuật toán Neighborhood CF (Item-CF):**
   - Item-CF sử dụng kỹ thuật co-occurrence với Top-50 láng giềng. Dù dữ liệu thưa, chỉ cần 2 bài hát cùng xuất hiện trong một vài danh sách nghe của cùng người dùng, độ tương đồng Cosine đã có thể tính toán chính xác, giúp thuật toán hoạt động linh hoạt ngay cả ở các cụm dữ liệu cục bộ.

---

## 3. THIÊN VỊ ĐỘ PHỔ BIẾN VÀ ĐỘ ĐA DẠNG GỢI Ý (POPULARITY BIAS & GINI COEFFICIENT)

Thiên vị độ phổ biến (Popularity Bias) là rủi ro kinh điển trong hệ thống gợi ý: Hệ thống chỉ chăm chăm gợi ý các bài hát thịnh hành toàn cầu (top hits), làm nghèo nàn trải nghiệm khám phá và triệt tiêu đuôi dài (long-tail items).

### Bảng đo lường độ phủ và chỉ số Gini trên tập Test:

| Mô hình | Độ phủ Catalog (Catalog Coverage) | Số bài hát được gợi ý ($\ge 1$ lần) | Chỉ số Gini (Gini Index) | Mức độ thiên vị độ phổ biến |
|---|:---:|:---:|:---:|---|
| **Popularity Baseline** | **1.00%** | 27 / 2.706 bài | **0.9959** | Cực kỳ tập trung (Extreme Bias) |
| **Implicit Neural CF** | 21.03% | 569 / 2.706 bài | 0.9753 | Rất cao (High Bias) |
| **BPR-MF** | 45.82% | 1.240 / 2.706 bài | 0.9065 | Trung bình - Cao |
| **Implicit ALS** | 62.90% | 1.702 / 2.706 bài | 0.8166 | Trung bình |
| **User-Based CF** | 65.45% | 1.771 / 2.706 bài | 0.8613 | Trung bình |
| **Item-Based CF** | **93.20%** | **2.522 / 2.706 bài** | **0.5661** | **Rất thấp / Phân tán tối ưu (Healthy Long-Tail)** |

```
Catalog Coverage (%):
  Item-Based CF : ███████████████████████ 93.20%
  User-Based CF : ████████████████        65.45%
  Implicit ALS  : ███████████████         62.90%
  BPR-MF        : ███████████             45.82%
  NeuMF         : █████                   21.03%
  Popularity    : ▌                        1.00%
```

### Phát hiện cốt lõi:
- **Item-Based CF đạt độ phủ danh mục kỷ lục (93.20%) và chỉ số Gini thấp nhất (0.5661).** Thuật toán gợi ý tới **2.522 bài hát khác nhau** trong tổng số 2.706 bài hát của danh mục. Điều này có nghĩa mỗi người dùng nhận được một danh sách Top-10 thực sự đặc trưng cho gu âm nhạc cá nhân của họ, thay vì nhận một danh sách bị chi phối bởi các bản nhạc "quốc dân".
- Ngược lại, **NeuMF** có chỉ số Gini rất cao (0.9753) và chỉ phủ 21.03% danh mục. Hàm kích hoạt phi tuyến và hàm loss BCE (Binary Cross-Entropy) có xu hướng kéo các bài hát có nhiều tương tác về phía có xác suất cao cho mọi người dùng.

---

## 4. PHÂN TẦNG NGƯỜI DÙNG THEO MỨC ĐỘ HOẠT ĐỘNG (USER ACTIVITY STRATIFICATION)

Chúng tôi chia $2.520$ người dùng trong tập Test thành 3 tầng dựa trên số lượng tương tác trong tập Train:
- **Tầng Thấp (Low-activity):** $\le 10$ bài hát trong Train ($N = 1,059$ người dùng, chiếm $42.0\%$).
- **Tầng Trung bình (Medium-activity):** $11 - 25$ bài hát trong Train ($N = 1,117$ người dùng, chiếm $44.3\%$).
- **Tầng Cao (High-activity):** $> 25$ bài hát trong Train ($N = 344$ người dùng, chiếm $13.7\%$).

### Bảng hiệu năng phân tầng (Hit Rate@10 & NDCG@10):

| Mô hình | HR@10 (Low) | NDCG@10 (Low) | HR@10 (Med) | NDCG@10 (Med) | HR@10 (High) | NDCG@10 (High) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Popularity** | 0.0529 | 0.0255 | 0.1101 | 0.0374 | 0.2122 | 0.0522 |
| **Implicit Neural CF** | 0.0604 | 0.0333 | 0.1558 | 0.0597 | 0.2936 | 0.0799 |
| **BPR-MF** | 0.0746 | 0.0440 | 0.1952 | 0.0807 | 0.3401 | 0.0960 |
| **Implicit ALS** | 0.1398 | 0.0745 | 0.2426 | 0.0982 | 0.3285 | 0.0791 |
| **User-Based CF** | **0.1464** | **0.0899** | 0.2641 | 0.1162 | 0.4419 | 0.1333 |
| **Item-Based CF** | 0.1369 | 0.0819 | **0.2766** | **0.1229** | **0.4738** | **0.1439** |

### Phân tích chi tiết từng nhóm người dùng:
1. **Đối với người dùng ít tương tác (Low-activity, $N=1,059$):**
   - Đây là bài toán khó nhất do hiệu ứng khởi đầu lạnh (Cold-start effect).
   - **User-Based CF** đạt điểm số cao nhất trong nhóm này (HR@10 = 0.1464, NDCG@10 = 0.0899), nhỉnh hơn một chút so với Item-CF (HR@10 = 0.1369). Lý do: Khi một người dùng mới chỉ nghe một vài bài hát, việc tìm kiếm những người dùng khác có cùng sở thích đối với một trong các bài hát đó (User-CF) mang lại tín hiệu gợi ý rộng hơn so với việc chỉ đối sánh các bài hát liên kết cục bộ.
   - Cả User-CF và Item-CF đều vượt xa Popularity Baseline (HR@10 = 0.0529) tới $+176.7\%$.
2. **Đối với người dùng có mức tương tác trung bình (Medium-activity, $N=1,117$):**
   - **Item-Based CF** vươn lên dẫn đầu với HR@10 = 0.2766 và NDCG@10 = 0.1229. Khi người dùng đã có từ 11 đến 25 bài hát, danh mục láng giềng theo bài hát trở nên cực kỳ phong phú và chính xác.
3. **Đối với người dùng tích cực (High-activity, $N=344$):**
   - **Item-Based CF** đạt hiệu năng xuất sắc: **HR@10 = 0.4738** (gần 1 trong 2 người dùng tích cực tìm thấy bài hát trúng đích trong Top-10) và NDCG@10 = 0.1439.
   - Điều này khẳng định: Càng nhiều dữ liệu tương tác lịch sử, Item-CF càng phát huy tối đa sức mạnh lập hồ sơ sở thích âm nhạc.

---

## 5. ĐỘ TRỄ SUY LUẬN VÀ KHẢ NĂNG MỞ RỘNG (INFERENCE LATENCY & SCALABILITY)

### Đo lường thời gian thực thi (Hardware: CPU đa nhân trên Windows):

| Mô hình | Thời gian Test toàn bộ ($2.520$ users) | Độ trễ trung bình (ms/user) | Thông lượng (Users/giây) | Dung lượng Checkpoint (KB) |
|---|:---:|:---:|:---:|:---:|
| **Popularity** | 2.46s | **0.97 ms** | **1,025.9 qps** | 1,184.2 KB |
| **User-Based CF** | 3.35s | **1.33 ms** | **752.1 qps** | 3,090.1 KB |
| **Implicit Neural CF** | 9.51s | 3.77 ms | 264.9 qps | 2,572.3 KB |
| **Implicit ALS** | 12.37s | 4.91 ms | 203.7 qps | 1,877.3 KB |
| **BPR-MF** | 13.45s | 5.34 ms | 187.4 qps | 1,877.5 KB |
| **Item-Based CF** | 23.54s | 9.34 ms | 107.1 qps | 4,033.8 KB |

### Đánh giá kỹ thuật hạ tầng:
- Cả 6 mô hình đều có độ trễ suy luận $< 10$ ms/user trên CPU. Điều này bảo đảm khi tích hợp vào FastAPI tại Phase M4, API endpoint `/api/v1/music/recommend` sẽ dễ dàng đạt SLA (Service Level Agreement) $< 50$ ms tính cả thời gian truyền mạng và serialize JSON.
- **User-Based CF** có tốc độ suy luận nhanh (1.33 ms) nhờ phép nhân ma trận thưa tối ưu `sub_matrix.T.dot(weights)`. Tuy nhiên, về mặt bộ nhớ dài hạn, kích thước ma trận người dùng tỷ lệ với $O(|U|^2)$, do đó khi mở rộng lên hàng triệu người dùng, **Item-Based CF** (kích thước cố định theo số bài hát $O(|I| \cdot K)$) hoặc **Implicit ALS** (vector dot product $O(|I| \cdot d)$) có tính mở rộng (scalability) bền vững hơn nhiều.

---

## 6. CÁC GIỚI HẠN CỦA TẬP DỮ LIỆU VÀ ĐỀ XUẤT NGHIÊN CỨU TƯƠNG LAI

1. **Giới hạn lấy mẫu tập dữ liệu (Dataset Subsampling Constraint):**
   - Bộ dữ liệu hiện tại được lọc 10-core từ tập Taste Profile của Million Song Dataset, giới hạn ở 2.520 người dùng và 2.706 bài hát nhằm bảo đảm huấn luyện trơn tru trong môi trường máy cá nhân. Mặc dù tập dữ liệu này hoàn toàn chuẩn hóa và bảo đảm tính hợp lệ khoa học, một số nghệ sĩ hoặc bài hát cực hiếm đã bị lược bỏ trong quá trình lọc 10-core.
2. **Thiếu thông tin đa phương thức (Lack of Audio Content & Lyrics):**
   - Tất cả 6 mô hình hiện tại đều là Collaborative Filtering (dựa trên tương tác ngầm định). Hệ thống chưa tận dụng các đặc trưng âm thanh như sóng âm (audio spectrogram), tiết tấu (tempo), giọng (pitch) hoặc lời bài hát (lyrics).
3. **Hướng phát triển tương lai (Future Directions):**
   - Tại các giai đoạn sau của hệ thống multi-domain, có thể nghiên cứu mô hình lai (Hybrid Recommender) kết hợp độ tương đồng Item-CF với embedding âm thanh từ mô hình trích xuất đặc trưng âm nhạc tự động (như MusiCNN hoặc CLMR) để giải quyết triệt để bài toán khởi đầu lạnh đối với bài hát mới phát hành chưa có lượt nghe.
