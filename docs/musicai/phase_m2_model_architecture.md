# KIẾN TRÚC VÀ THUẬT TOÁN 6 MÔ HÌNH GỢI Ý BÀI HÁT
## PHASE M2: MUSICAI 6-MODEL ARCHITECTURE SPECIFICATION

**Dự án:** Multi-Domain AI Recommendation System (CineAI V2.0 & MusicAI)  
**Nhánh Git:** `feature/musicai-models`  
**Vai trò:** Senior Machine Learning Engineer & Recommender Systems Researcher  
**Thực thể gợi ý:** **Bài hát (`song_id`)**  
**Ngày hoàn thành:** 28/10/2026  

---

## 1. TỔNG QUAN HỆ THỐNG 6 MÔ HÌNH MUSICAI

Khung mô hình MusicAI được thiết kế theo hướng đối tượng hướng tới phản hồi ngầm định (Implicit Feedback), kế thừa từ lớp cơ sở [`BaseMusicRecommender`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/music/models/base_music_recommender.py) với 6 thuật toán đại diện cho 4 trường phái RecSys lớn:

```
                          [ BỘ 6 MÔ HÌNH MUSICAI ]
                                     │
      ┌──────────────────┬───────────┴───────────┬──────────────────┐
      ▼                  ▼                       ▼                  ▼
[ BASELINE ]       [ LÂN CẬN (CF) ]       [ MATRIX FACTOR ]   [ DEEP LEARNING ]
1. Popularity      2. Item-Based CF       4. Implicit ALS     6. Implicit NeuMF
                   3. User-Based CF       5. BPR-MF (Pairwise)
```

---

## 2. CHI TIẾT THUẬT TOÁN TỪNG MÔ HÌNH

### 2.1. Model 1 – Popularity Baseline (`PopularityMusicRecommender`)
- **Tập tin mã nguồn:** [`src/music/models/popularity.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/music/models/popularity.py)
- **Ý tưởng:** Xếp hạng các bài hát dựa trên mức độ phủ sóng người nghe và tổng lượt nghe tích lũy trong tập Train.
- **Công thức tính điểm:**
  $$\text{Score}(i) = \ln(1 + \text{distinct\_listeners}_i) \cdot \ln(1 + \text{total\_plays}_i)$$
- **Chiến lược lọc & Fallback:**
  - Tự động loại trừ các bài hát người dùng đã nghe trong tập Train khi gợi ý (`exclude_seen=True`).
  - Người dùng mới (Cold User) được gợi ý danh sách Top bài hát phổ biến nhất toàn cầu.
- **Độ phức tạp suy luận:** $O(1)$ tra cứu mảng đã sắp xếp trước.

---

### 2.2. Model 2 – Item-Based Collaborative Filtering (`ItemCFMusicRecommender`)
- **Tập tin mã nguồn:** [`src/music/models/item_cf.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/music/models/item_cf.py)
- **Ý tưởng:** Tính toán độ tương đồng giữa các bài hát dựa trên véc-tơ người nghe chung trên ma trận thưa $2,706 \times 2,520$.
- **Độ đo tương đồng:** Cosine Similarity có ngưỡng lọc láng giềng $\text{sim} \ge 0.01$ và giới hạn Top-$K=50$ láng giềng gần nhất cho mỗi bài hát.
- **Dự đoán sở thích:**
  $$\hat{s}(u, i) = \sum_{j \in \mathcal{H}_u \cap \mathcal{N}_{50}(i)} \text{sim}(i, j) \cdot c_{uj}$$
  với hàm độ tin cậy ngầm định: $c_{uj} = 1 + \alpha \cdot \ln(1 + r_{uj})$ ($\alpha = 40.0$).
- **Xử lý biên:** Nếu bài hát ứng viên không có láng giềng nào giao với lịch sử người dùng, điểm số trả về là 0.0 hoặc điểm phổ biến thu nhỏ.

---

### 2.3. Model 3 – User-Based Collaborative Filtering (`UserCFMusicRecommender`)
- **Tập tin mã nguồn:** [`src/music/models/user_cf.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/music/models/user_cf.py)
- **Ý tưởng:** Tìm kiếm Top-$K=50$ người dùng có gu âm nhạc tương đồng nhất qua độ đo Cosine trên véc-tơ độ tin cậy nghe nhạc $2,520 \times 2,706$.
- **Dự đoán sở thích:**
  $$\hat{s}(u, i) = \sum_{v \in \mathcal{N}_{50}(u)} \text{sim}(u, v) \cdot c_{vi}$$
  Tối ưu hóa tính toán qua phép nhân ma trận thưa `sub_matrix.T.dot(neighbor_weights)`.
- **Fallback:** Nếu người dùng không có láng giềng phù hợp, chuyển hướng fallback sang Popularity.

---

### 2.4. Model 4 – Implicit Alternating Least Squares (`IALSMusicRecommender`)
- **Tập tin mã nguồn:** [`src/music/models/ials.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/music/models/ials.py)
- **Công trình gốc:** Hu, Koren & Volinsky (IEEE ICDM 2008).
- **Hàm mất mát bình phương có trọng số:**
  $$\mathcal{L} = \sum_{u, i} c_{ui} (p_{ui} - \mathbf{x}_u^T \mathbf{y}_i)^2 + \lambda \left(\sum_u \|\mathbf{x}_u\|_2^2 + \sum_i \|\mathbf{y}_i\|_2^2\right)$$
  với $p_{ui} = 1$ khi $r_{ui} \ge 1$ và $c_{ui} = 1 + \alpha \ln(1 + r_{ui})$.
- **Giải thuật cập nhật ma trận Gram đóng $O(k^3)$:**
  $$\mathbf{x}_u = \left( Y^T Y + Y_u^T (C^u - I) Y_u + \lambda I_k \right)^{-1} Y_u^T C^u \mathbf{p}_u$$
  $$\mathbf{y}_i = \left( X^T X + X_i^T (C^i - I) X_i + \lambda I_k \right)^{-1} X_i^T C^i \mathbf{p}_i$$
- **Tham số tối ưu:** $k=32$ factors, $\lambda=0.1$, $\alpha=40.0$, 15 iterations.

---

### 2.5. Model 5 – Bayesian Personalized Ranking (`BPRMFMusicRecommender`)
- **Tập tin mã nguồn:** [`src/music/models/bpr_mf.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/music/models/bpr_mf.py)
- **Công trình gốc:** Rendle, Freudenthaler, Gantner, Schmidt-Thieme (UAI 2009).
- **Hàm mục tiêu xếp hạng cặp (Pairwise AUC Optimization):**
  $$\max_{W, H} \sum_{(u, i, j) \in D_S} \ln \sigma(\mathbf{w}_u^T \mathbf{h}_i - \mathbf{w}_u^T \mathbf{h}_j) - \frac{\lambda}{2} \left( \|\mathbf{w}_u\|^2 + \|\mathbf{h}_i\|^2 + \|\mathbf{h}_j\|^2 \right)$$
  với $i \in \mathcal{H}_u^{\text{train}}$ và $j \notin \mathcal{H}_u^{\text{train}}$.
- **Huấn luyện Mini-batch SGD:** Sử dụng cập nhật vector hóa kết hợp `np.add.at` chống xung đột bộ nhớ, kích thước batch 1024, học suất $\eta = 0.05$, 20 epochs, seed cố định 42.

---

### 2.6. Model 6 – Implicit Neural Collaborative Filtering (`ImplicitNeuralCFRecommender`)
- **Tập tin mã nguồn:** [`src/music/models/implicit_neural_cf.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/music/models/implicit_neural_cf.py)
- **Cấu trúc mạng sâu PyTorch (NeuMF):**
  - **Nhánh GMF:** Nhúng người dùng và bài hát $d_G=32$, tích Hadamard $\phi^{\text{GMF}} = \mathbf{p}_u^G \odot \mathbf{q}_i^G$.
  - **Nhánh MLP:** Nhúng người dùng và bài hát $d_M=32$, ghép nối $[\mathbf{p}_u^M, \mathbf{q}_i^M] \in \mathbb{R}^{64}$. Đi qua 3 tầng nơ-ron: $64 \to 32 \to 16$ với BatchNorm1d, ReLU và Dropout(0.2).
  - **Tầng kết hợp (Fusion Layer):** Ghép $[\phi^{\text{GMF}}, \phi^{\text{MLP}}] \in \mathbb{R}^{48} \to 1$.
- **Hàm mất mát & Lấy mẫu âm:** `BCEWithLogitsLoss()` với tỷ lệ lấy mẫu âm $1:4$ (1 tương tác dương đi kèm 4 mẫu âm unobserved ngẫu nhiên tái tạo theo từng epoch).

---

## 3. GIAO DIỆN HỢP ĐỒNG LẬP TRÌNH BẢO ĐẢM TÍNH NHẤT QUÁN

Mọi mô hình đều thực hiện đúng hợp đồng trừu tượng:
- `fit(train_df: pd.DataFrame) -> BaseMusicRecommender`
- `recommend(user_id: str, top_k: int = 10, exclude_seen: bool = True) -> List[Tuple[str, float]]`
- `score(user_id: str, song_ids: Sequence[str]) -> np.ndarray`
- `save(path: Union[str, Path]) -> None`
- `load(path: Union[str, Path]) -> BaseMusicRecommender`
- Đảm bảo 100% không trả về NaN, Inf hay bài hát đã nghe trong tập Train.
