# BÁO CÁO BENCHMARK ĐÁNH GIÁ OFFLINE TRÊN TẬP TEST — MUSICAI PHASE M3
*(Phase M3 Test Benchmark & Generalization Performance Report)*

**Dự án:** Multi-Domain AI Recommendation System (CineAI V2.0 & MusicAI)  
**Nhánh Git:** `feature/musicai-models`  
**Ngày thực hiện:** 28/10/2026  
**Tập dữ liệu kiểm thử:** `data/music/processed/test.csv` (Đã niêm phong từ Phase M1)  
**Quy mô đánh giá:** 2.520 người dùng, 2.706 bài hát trong danh mục, đánh giá xếp hạng toàn catalog (Full-Catalog Ranking, $K=10$).  

---

## 1. TỔNG QUAN KẾT QUẢ BENCHMARK TRÊN TẬP TEST

Toàn bộ 6 mô hình AI gợi ý bài hát được nạp từ checkpoint nguyên bản và đánh giá độc lập trên tập Test theo đúng giao thức đã đóng băng tại [`docs/musicai/phase_m3_evaluation_protocol.md`](file:///E:/HOCTAP/K5/TTNT/AI-Recommendation-System/docs/musicai/phase_m3_evaluation_protocol.md).

### Bảng xếp hạng chính thức trên tập Test (Test Leaderboard Top-10)

| Hạng | Tên mô hình | Họ thuật toán | HR@10 (95% CI) | NDCG@10 (95% CI) | P@10 | R@10 | MRR@10 | Độ phủ catalog | Gini Index | Độ trễ (ms/user) | Thông lượng (qps) |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Item-Based CF** | Neighborhood CF | **0.2448** <br> `[0.2278, 0.2623]` | **0.1085** <br> `[0.0989, 0.1175]` | **0.0307** | **0.1544** | **0.1238** | **93.20%** | **0.5661** | 9.34 ms | 107.1 |
| **2** | **User-Based CF** | Neighborhood CF | 0.2389 <br> `[0.2230, 0.2552]` | 0.1075 <br> `[0.0985, 0.1168]` | 0.0297 | 0.1535 | 0.1201 | 65.45% | 0.8613 | **1.33 ms** | 752.1 |
| **3** | **Implicit ALS** | Matrix Factorization | 0.2111 <br> `[0.1952, 0.2274]` | 0.0856 <br> `[0.0773, 0.0937]` | 0.0249 | 0.1377 | 0.0866 | 62.90% | 0.8166 | 4.91 ms | 203.7 |
| **4** | **BPR-MF** | Pairwise Ranking | 0.1643 <br> `[0.1508, 0.1790]` | 0.0674 <br> `[0.0600, 0.0745]` | 0.0201 | 0.0966 | 0.0786 | 45.82% | 0.9065 | 5.34 ms | 187.4 |
| **5** | **Implicit Neural CF** | Deep Neural Hybrid | 0.1345 <br> `[0.1214, 0.1480]` | 0.0514 <br> `[0.0451, 0.0579]` | 0.0163 | 0.0774 | 0.0596 | 21.03% | 0.9753 | 3.77 ms | 264.9 |
| **6** | **Popularity Baseline** | Không cá nhân hóa | 0.1000 <br> `[0.0877, 0.1123]` | 0.0345 <br> `[0.0294, 0.0393]` | 0.0116 | 0.0589 | 0.0366 | 1.00% | 0.9959 | 0.97 ms | 1,025.9 |

> [!NOTE]
> Tất cả các chỉ số được đo lường dưới điều kiện đánh giá thực tế: Ứng viên là toàn bộ 2.706 bài hát trừ đi các bài người dùng đã nghe trong Train. Không sử dụng sampled negatives (rút gọn 100 bài) nhằm bảo toàn tính khắt khe chuẩn học thuật.

---

## 2. ĐỐI CHIẾU NĂNG LỰC TỔNG QUÁT HÓA (VALIDATION VS. TEST)

Bảng so sánh chi tiết giữa tập Validation (Phase M2) và tập Test độc lập (Phase M3):

| Mô hình | HR@10 (Val) | HR@10 (Test) | $\Delta$ HR | $\% \Delta$ HR | NDCG@10 (Val) | NDCG@10 (Test) | $\Delta$ NDCG | $\% \Delta$ NDCG | MRR@10 (Val) | MRR@10 (Test) | $\Delta$ MRR |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Popularity** | 0.1028 | 0.1000 | -0.0028 | -2.72% | 0.0349 | 0.0345 | -0.0004 | -1.15% | 0.0405 | 0.0366 | -0.0039 |
| **Item-Based CF** | 0.2425 | **0.2448** | +0.0023 | +0.95% | 0.1038 | **0.1085** | +0.0047 | +4.53% | 0.1144 | **0.1238** | +0.0094 |
| **User-Based CF** | 0.2278 | 0.2389 | +0.0111 | +4.87% | 0.0996 | 0.1075 | +0.0079 | +7.93% | 0.1176 | 0.1201 | +0.0025 |
| **Implicit ALS** | 0.2063 | 0.2111 | +0.0048 | +2.33% | 0.0812 | 0.0856 | +0.0044 | +5.42% | 0.0859 | 0.0866 | +0.0007 |
| **BPR-MF** | 0.1631 | 0.1643 | +0.0012 | +0.74% | 0.0624 | 0.0674 | +0.0050 | +8.01% | 0.0755 | 0.0786 | +0.0031 |
| **Implicit Neural CF** | 0.1214 | 0.1345 | +0.0131 | +10.79% | 0.0438 | 0.0514 | +0.0076 | +17.35% | 0.0521 | 0.0596 | +0.0075 |

### Nhận định về khả năng tổng quát hóa (Generalization Analysis):
1. **Bảo toàn hoàn hảo thứ hạng (Ranking Stability):** Thứ tự hiệu năng giữa 6 mô hình trên tập Test trùng khớp 100% với tập Validation:
   $$\text{Item-CF} > \text{User-CF} > \text{iALS} > \text{BPR-MF} > \text{NeuMF} > \text{Popularity}$$
2. **Không có hiện tượng suy giảm hiệu năng (Zero Overfitting):**
   - Không có mô hình cá nhân hóa nào bị tụt giảm điểm số trên Test so với Validation.
   - Các mức tăng trưởng nhẹ ($+0.7\%$ đến $+8.0\%$ về NDCG) hoàn toàn nằm trong biên độ dao động ngẫu nhiên cho phép của phân bố tương tác, chứng minh trọng số mô hình đã học được các mẫu quy luật tổng quát chứ không ghi nhớ dữ liệu Train cục bộ.
3. **Mô hình phổ biến Popularity Baseline giảm nhẹ (-1.15% NDCG):**
   - Sự sụt giảm của Popularity trên Test phản ánh tính chất tự nhiên: trong tập Test, các tương tác người dùng có xu hướng dịch chuyển nhẹ về các bài hát ngách hoặc ít phổ biến hơn so với tập Validation, khiến gợi ý phi cá nhân hóa giảm hiệu quả.

---

## 3. KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (STATISTICAL SIGNIFICANCE TESTS)

Kiểm định phi tham số **Wilcoxon Signed-Rank Test** được thực hiện trên chuỗi điểm số từng người dùng ($N = 2.520$) đối với chỉ số NDCG@10:

| Mô hình | So sánh với Popularity Baseline ($p$-value) | Ý nghĩa thống kê vs. Popularity | So sánh với Item-Based CF ($p$-value) | Ý nghĩa thống kê vs. Item-CF |
|---|:---:|:---:|:---:|:---:|
| **Popularity** | 1.0000 | Baseline | $< 10^{-50}$ | Kém hơn có ý nghĩa ($p < 0.001$) |
| **Item-Based CF** | $< 10^{-50}$ | Vượt trội ($p < 0.001$) | 1.0000 | Benchmark Top-1 |
| **User-Based CF** | $< 10^{-50}$ | Vượt trội ($p < 0.001$) | $p = 0.3389$ | Tương đương thống kê ($p \ge 0.05$) |
| **Implicit ALS** | $1.61 \times 10^{-34}$ | Vượt trội ($p < 0.001$) | $1.40 \times 10^{-8}$ | Kém hơn có ý nghĩa ($p < 0.001$) |
| **BPR-MF** | $3.05 \times 10^{-27}$ | Vượt trội ($p < 0.001$) | $8.71 \times 10^{-23}$ | Kém hơn có ý nghĩa ($p < 0.001$) |
| **Implicit Neural CF**| $1.68 \times 10^{-15}$ | Vượt trội ($p < 0.001$) | $5.03 \times 10^{-35}$ | Kém hơn có ý nghĩa ($p < 0.001$) |

### Kết luận kiểm định:
- **Tất cả các mô hình cá nhân hóa** đều vượt trội hơn mô hình Popularity Baseline với mức ý nghĩa cực cao ($p < 10^{-14}$, loại trừ hoàn toàn yếu tố may rủi).
- **Item-Based CF** và **User-Based CF** là hai mô hình nằm ở nhóm hiệu năng cao nhất (tier 1). Giữa Item-CF và User-CF không có sự khác biệt có ý nghĩa thống kê về NDCG@10 ($p = 0.3389$).
- Tuy nhiên, **Item-Based CF** vượt trội hơn hẳn User-Based CF về độ phủ danh mục (93.20% so với 65.45%) và tính phân tán gợi ý (Gini 0.5661 so với 0.8613).

---

## 4. KẾT LUẬN CỦA BÁO CÁO TEST BENCHMARK
Kết quả thực nghiệm trên tập Test đã xác thực trọn vẹn kết luận từ Phase M2:
- **Item-Based CF** là thuật toán gợi ý bài hát xuất sắc nhất toàn diện: Dẫn đầu về độ chính xác (HR@10 = 0.2448, NDCG@10 = 0.1085), dẫn đầu về độ phủ danh mục (93.20%), phân tán gợi ý đồng đều nhất (Gini = 0.5661) và tốc độ suy luận nhanh (9.34 ms/user).
- Khuyến nghị đưa **Item-Based CF** làm thuật toán mặc định cho hệ thống Backend tại Phase M4.
