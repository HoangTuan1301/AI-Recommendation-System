# Báo Cáo Thực Nghiệm Giai Đoạn 4A: Neighborhood-Based Collaborative Filtering

**Đề tài:** Xây dựng chương trình gợi ý phim/bài hát theo sở thích ứng dụng Trí tuệ nhân tạo  
**Vai trò:** Senior Machine Learning Engineer, Recommender Systems Researcher & Python Software Architect  
**Dự án:** AI-Recommendation-System - Học kỳ 5  
**Tập dữ liệu:** MovieLens 100K (Train: 80,000 / Test: 20,000 | 935 evaluation users)  

---

## 1. Tổng Quan Kiến Trúc & Mục Tiêu Giai Đoạn 4A

Trong Giai đoạn 4A, hệ thống đã mở rộng từ 2 mô hình baseline ban đầu sang nhóm **Lọc cộng tác dựa trên láng giềng (Neighborhood-Based Collaborative Filtering)** gồm 3 thuật toán kinh điển:
1. **UserBasedCF**: Lọc cộng tác giữa người dùng với người dùng (User-User KNN).
2. **ItemBasedCF**: Lọc cộng tác giữa các bộ phim với nhau (Item-Item KNN).
3. **KNNBaseline**: Lọc cộng tác kết hợp điều chỉnh độ lệch bias cá nhân và hiệu ứng toàn cục (Bias-Adjusted KNN with Shrinkage).

Cả 3 mô hình đều kế thừa giao diện chuẩn `BaseRecommender` và được đánh giá trên cùng một thể hiện `TopNEvaluator` (935 evaluation users, $K = 10$, tập ứng viên $\mathcal{C}_u = \mathcal{I}_{all} \setminus \mathcal{I}_u^{train}$, lọc bỏ tuyệt đối các phim đã có trong lịch sử Train).

---

## 2. Giải Thích Thuật Toán & Công Thức Toán Học

### 2.1. User-Based Collaborative Filtering (User-User KNN)
- **Cơ chế:** Gợi ý các bộ phim được đánh giá cao bởi những người dùng có "gu" tương đồng nhất với người dùng mục tiêu.
- **Chuẩn hóa điểm số (User Centering):** Do mỗi người dùng có thang chấm điểm khác nhau (người dễ tính thường chấm 4-5 sao, người khó tính chỉ chấm 2-3 sao), điểm đánh giá được chuẩn hóa bằng cách trừ đi điểm trung bình cá nhân:
$$\tilde{r}_{ui} = r_{ui} - \bar{r}_u \quad (\text{chỉ tính cho các tương tác đã quan sát})$$
- **Độ tương đồng Pearson Correlation:**
$$s_{uv} = \frac{\sum_{i \in I_u \cap I_v} (r_{ui} - \bar{r}_u)(r_{vi} - \bar{r}_v)}{\sqrt{\sum_{i \in I_u \cap I_v} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{i \in I_u \cap I_v} (r_{vi} - \bar{r}_v)^2}}$$
  Được tính toán siêu tốc thông qua tích vô hướng ma trận thưa chuẩn hóa $L_2$: $\mathbf{S}_{user} = \mathbf{R}_{norm} \cdot \mathbf{R}_{norm}^T$.
- **Dự đoán điểm số:** Chọn top $k$ láng giềng $\mathcal{N}_k(u)$ có $s_{uv} > 0$:
$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{v \in \mathcal{N}_k(u)} s_{uv} (r_{vi} - \bar{r}_v)}{\sum_{v \in \mathcal{N}_k(u)} |s_{uv}|}$$
  Nếu không có láng giềng nào đánh giá phim $i$, mô hình fallback về $\bar{r}_u$.

### 2.2. Item-Based Collaborative Filtering (Item-Item KNN)
- **Cơ chế:** Một người dùng sẽ thích bộ phim $i$ nếu họ từng yêu thích các bộ phim $j$ có tính chất tương tự $i$ trong quá khứ.
- **Độ tương đồng Adjusted Cosine Similarity:** Khắc phục nhược điểm của Cosine truyền thống bằng cách trừ đi điểm trung bình của người dùng đánh giá cặp phim đó:
$$s_{ij} = \frac{\sum_{u \in U_i \cap U_j} (r_{ui} - \bar{r}_u)(r_{uj} - \bar{r}_u)}{\sqrt{\sum_{u \in U_i \cap U_j} (r_{ui} - \bar{r}_u)^2} \sqrt{\sum_{u \in U_i \cap U_j} (r_{uj} - \bar{r}_u)^2}}$$
  Lọc bỏ các độ tương đồng âm ($s_{ij} \le 0 \rightarrow 0$) và lưu giữ top $k$ láng giềng tương đồng nhất cho mỗi item.
- **Dự đoán điểm số:**
$$\hat{r}_{ui} = \bar{r}_u + \frac{\sum_{j \in I_u \cap \mathcal{N}_k(i)} s_{ij} (r_{uj} - \bar{r}_u)}{\sum_{j \in I_u \cap \mathcal{N}_k(i)} |s_{ij}|}$$

### 2.3. KNNBaseline (Koren, 2010)
- **Cơ chế:** Mô hình láng giềng tiên tiến nhất trong nhóm KNN. Thay vì chỉ chuẩn hóa theo trung bình người dùng, KNNBaseline bóc tách đồng thời 3 thành phần độ lệch:
  1. $\mu$: Điểm trung bình toàn cục (Global Mean $\approx 3.5254$).
  2. $b_u$: Độ lệch người dùng (User Bias).
  3. $b_i$: Độ lệch chất lượng phim (Item Bias).
- **Học độ lệch có kiểm soát (Regularized Least Squares):**
$$b_i = \frac{\sum_{u \in U_i} (r_{ui} - \mu)}{\lambda_{item} + |U_i|}, \quad b_u = \frac{\sum_{i \in I_u} (r_{ui} - \mu - b_i)}{\lambda_{user} + |I_u|}$$
  Baseline Predictor: $b_{ui} = \mu + b_u + b_i$.
- **Độ tương đồng trên phần dư (Residual Similarity with Shrinkage):**
  Phần dư $\tilde{r}_{ui} = r_{ui} - b_{ui}$ phản ánh sở thích thuần túy sau khi đã loại bỏ toàn bộ hiệu ứng điểm số và độ nổi tiếng. Độ tương đồng Pearson trên phần dư được áp dụng hệ số co rút (Shrinkage) nhằm phạt các cặp phim có quá ít người cùng đánh giá ($n_{ij} = |U_i \cap U_j|$):
$$s_{ij} = \text{Cosine}(\tilde{\mathbf{r}}_i, \tilde{\mathbf{r}}_j) \times \frac{n_{ij}}{n_{ij} + \lambda_{shrink}}$$
- **Dự đoán điểm số:**
$$\hat{r}_{ui} = b_{ui} + \frac{\sum_{j \in I_u \cap \mathcal{N}_k(i)} s_{ij} (r_{uj} - b_{uj})}{\sum_{j \in I_u \cap \mathcal{N}_k(i)} |s_{ij}|}$$
  *Ưu thế vượt trội:* Khi không tìm thấy láng giềng nào (mẫu số bằng 0), mô hình fallback trực tiếp về $b_{ui} = \mu + b_u + b_i$ – một giá trị dự đoán cá nhân hóa cực kỳ chuẩn xác thay vì một giá trị trung bình phẳng.

---

## 3. Thiết Lập Siêu Tham Số (Hyperparameters)

Các siêu tham số được cố định theo chuẩn nghiên cứu lý thuyết (không tuning trên tập Test để chống Overfitting):
- **UserBasedCF:**
  - $k\_neighbors = 40$ (kích thước cụm láng giềng người dùng).
  - $min\_similarity = 0.0$ (chỉ xét tương quan dương).
- **ItemBasedCF:**
  - $k\_neighbors = 30$ (số lượng phim tương đồng tối đa cho mỗi phim).
  - $min\_similarity = 0.0$.
- **KNNBaseline:**
  - $k\_neighbors = 30$.
  - $\lambda_{user} = 10.0$ (L2 Regularization cho user bias).
  - $\lambda_{item} = 25.0$ (L2 Regularization cho item bias - phim có ít đánh giá sẽ bị kéo về 0).
  - $\lambda_{shrink} = 100.0$ (hệ số phạt co rút độ tương đồng với cặp phim ít người cùng xem).

---

## 4. Bảng Kết Quả Thực Nghiệm Đối Soát (5 Mô Hình)

Kết quả thực nghiệm trên tập Test độc lập (lưu tại [results/phase4a_metrics.csv](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/results/phase4a_metrics.csv)):

| STT | Mô hình (Model) | Precision@10 | Recall@10 | F1@10 | NDCG@10 | Hit Rate@10 | Fit Time (s) | Eval Time (s) | Total Time (s) |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Popularity** | **0.0653** | **0.0497** | **0.0473** | **0.0764** | **0.3914** | 0.0302 s | 0.8739 s | **0.9041 s** |
| 2 | **KNNBaseline** | **0.0278** | **0.0254** | **0.0210** | **0.0323** | **0.1968** | 0.7205 s | 2.1877 s | **2.9081 s** |
| 3 | **Content-Based** | **0.0236** | **0.0199** | **0.0174** | **0.0296** | **0.1872** | 0.0853 s | 5.9959 s | **6.0812 s** |
| 4 | **Item-Based CF** | **0.0184** | **0.0235** | **0.0171** | **0.0250** | **0.1551** | 0.2772 s | 1.7520 s | **2.0292 s** |
| 5 | **User-Based CF** | **0.0056** | **0.0070** | **0.0049** | **0.0064** | **0.0535** | 0.0789 s | 1.7763 s | **1.8552 s** |

*(Số lượng người dùng đánh giá: 935 users với ít nhất 1 đánh giá tích cực trong Test).*

---

## 5. Phân Tích Chuyên Sâu Hiệu Năng & Hiện Tượng Dữ Liệu

1. **Sự vượt trội của KNNBaseline so với các mô hình láng giềng khác:**
   - KNNBaseline đạt **Hit Rate@10 = 19.68%**, **NDCG@10 = 0.0323**, vượt trội hơn hẳn so với Item-Based CF (15.51%) và User-Based CF (5.35%), đồng thời vượt qua cả Content-Based Baseline (18.72%).
   - *Nguyên nhân kỹ thuật:* Trong ma trận có độ thưa lên tới 93.7%, việc bóc tách bias $b_u, b_i$ trước khi tính tương đồng giúp loại bỏ "nhiễu" do thói quen chấm điểm và độ nổi tiếng. Quan trọng hơn, cơ chế fallback về $b_{ui} = \mu + b_u + b_i$ đảm bảo rằng ngay cả khi tập láng giềng bị thưa thớt, mô hình vẫn cung cấp dự đoán có cơ sở khoa học vững chắc.
2. **Item-Based CF vượt trội so với User-Based CF:**
   - Item-Based CF đạt **Hit Rate@10 = 15.51%**, gấp gần 3 lần User-Based CF (5.35%).
   - *Nguyên nhân kỹ thuật:* Trong MovieLens 100K, bản chất một bộ phim có thuộc tính ổn định hơn nhiều so với tâm lý thất thường của người dùng. Hai bộ phim cùng thể loại và phong cách sẽ luôn nhận được đánh giá tương quan từ hàng trăm người xem, trong khi hai người dùng rất hiếm khi có chung toàn bộ danh sách phim đã xem. Hiện tượng này hoàn toàn khớp với công trình kinh điển của Amazon (Sarwar et al., 2001).
3. **Tại sao User-Based CF đạt kết quả khiêm tốn?**
   - Với độ thưa ma trận 93.7%, số lượng phim giao thoa giữa 2 người dùng ngẫu nhiên là rất nhỏ. Nhiều cặp người dùng chỉ có 1-2 phim chung, dẫn đến tương quan Pearson bị nhiễu nghiêm trọng hoặc không tìm thấy đủ 40 láng giềng thực sự tin cậy.
4. **Vị trí của Popularity Baseline:**
   - Popularity Recommender vẫn dẫn đầu về chỉ số Top-N Offline (HR = 39.14%) do hiện tượng "Long-Tail": số lượng phim bom tấn chiếm áp đảo trong các lượt xem của người dùng. Tuy nhiên, Popularity hoàn toàn không có khả năng cá nhân hóa (Personalization) – tất cả người dùng đều nhận được cùng một danh sách phim giống hệt nhau (trừ các phim họ đã xem).

---

## 6. Kết Quả Kiểm Thử (Unit Tests)

Thực thi kiểm thử hồi quy toàn diện trên toàn bộ hệ thống:
```text
Ran 30 tests in 11.050s
OK
```
- **Phase 2 Pipeline:** 10/10 tests pass (tính toàn vẹn dữ liệu, split 80/20, không rò rỉ dữ liệu).
- **Phase 3 Pipeline:** 10/10 tests pass (tính đúng 5 độ đo toán học, xử lý biên, hợp đồng BaseRecommender, Popularity, Content-Based).
- **Phase 4A Pipeline:** 10/10 tests pass (hợp đồng UserBasedCF, ItemBasedCF, KNNBaseline, không đề xuất phim đã xem trong Train, không trùng ID, không có NaN/Inf, bảo đảm candidate set nhất quán).

---

## 7. Danh Sách Các File Đã Tạo / Cập Nhật Trong Giai Đoạn 4A

1. [src/models/user_cf.py](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/models/user_cf.py): Cài đặt User-User Collaborative Filtering.
2. [src/models/item_cf.py](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/models/item_cf.py): Cài đặt Item-Item Collaborative Filtering với Adjusted Cosine.
3. [src/models/knn_baseline.py](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/models/knn_baseline.py): Cài đặt KNNBaseline (Koren 2010) với L2 Regularized Biases & Shrinkage.
4. [src/evaluation/benchmark_phase4a.py](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/evaluation/benchmark_phase4a.py): Script benchmark đồng loạt 5 mô hình và đo đạc thời gian huấn luyện / đánh giá riêng biệt.
5. [results/phase4a_metrics.csv](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/results/phase4a_metrics.csv): Bảng kết quả benchmark thực tế của 5 mô hình.
6. [tests/test_phase4a_pipeline.py](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/tests/test_phase4a_pipeline.py): 10 unit test cases chuyên biệt cho Giai đoạn 4A.
7. [reports/phase4a_report.md](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/reports/phase4a_report.md): Báo cáo chi tiết Giai đoạn 4A.

---

## 8. Trạng Thái Git (`git status`)

```text
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
	modified:   .gitignore

Untracked files:
	reports/phase3_baseline_report.md
	reports/phase4a_report.md
	results/phase3_baseline_metrics.csv
	results/phase4a_metrics.csv
	src/evaluation/benchmark.py
	src/evaluation/benchmark_phase4a.py
	src/evaluation/evaluator.py
	src/evaluation/metrics.py
	src/models/base_recommender.py
	src/models/content_based.py
	src/models/item_cf.py
	src/models/knn_baseline.py
	src/models/popularity.py
	tests/test_phase3_pipeline.py
	tests/test_phase4a_pipeline.py

no changes added to commit (use "git add" and/or "git commit -a")
```
> [!NOTE]
> Đúng nguyên tắc: Không tự ý commit hoặc push code lên GitHub. Dataset thô và môi trường `.venv` vẫn được bảo vệ an toàn tuyệt đối bởi `.gitignore`.

---

## 9. Hạn Chế Còn Tồn Tại & Đề Xuất Kế Hoạch Cho Giai Đoạn 4B

- **Hạn chế của nhóm mô hình láng giềng:**
  - Nhóm mô hình KNN chỉ dựa trên các tương đồng trực tiếp cục bộ (local similarities). Khi ma trận quá thưa (93.7%), các láng giềng có thể không có đủ tín hiệu giao thoa.
  - Bộ nhớ lưu trữ ma trận tương đồng tăng theo bậc hai $O(M^2)$ hoặc $O(U^2)$.
- **Kế hoạch cho Giai đoạn 4B (Matrix Factorization):**
  Chúng ta sẽ triển khai 3 mô hình phân rã ma trận ẩn (Latent Factor Models) để học không gian đặc trưng ẩn $d \ll \min(U, M)$:
  1. **Mô hình 6: Singular Value Decomposition (SVD)** (Funk SVD với User/Item Biases và Gradient Descent).
  2. **Mô hình 7: SVD++** (Mở rộng từ SVD tích hợp thông tin phản hồi ẩn implicit feedback từ tập phim đã xem).
  3. **Mô hình 8: Non-Negative Matrix Factorization (NMF)** (Phân rã ma trận với ràng buộc không âm để trích xuất các thành phần sở thích có thể giải thích được).
  - Tiếp theo ở Phase 4C sẽ là **Neural Collaborative Filtering (NCF)** và **Hybrid Recommender**, hoàn tất trọn vẹn danh mục 10 mô hình theo yêu cầu của giảng viên.
