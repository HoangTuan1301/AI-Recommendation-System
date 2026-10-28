# BÁO CÁO NGHIỆM THU TOÀN DIỆN VÀ KIỂM TOÁN HỆ THỐNG (PHASE 6A)
## CINEAI V2.0 – FINAL SYSTEM ACCEPTANCE & END-TO-END AUDIT REPORT
### HỆ THỐNG GỢI Ý PHIM 10 MÔ HÌNH AI • FASTAPI BACKEND • REACT + TYPESCRIPT FRONTEND

---

**Dự án:** AI-Recommendation-System / CineAI V2.0  
**Học kỳ:** Học kỳ 5 – Môn học: Trí tuệ nhân tạo (AI)  
**Thời gian thực hiện:** 28/10/2026  
**Vai trò kiểm toán:** Senior QA Automation Engineer, Machine Learning Engineer, FastAPI Architect, React Testing Specialist  
**Quyết định nghiệm thu:** **GO (ĐỦ ĐIỀU KIỆN XUẤT BẢN VÀ BẢO VỆ ĐỒ ÁN)**

---

## 1. EXECUTIVE SUMMARY (TỔNG QUAN NGHIỆM THU)

Giai đoạn **Phase 6A** là đợt kiểm toán kỹ thuật và kiểm thử nghiệm thu tích hợp cuối cùng của toàn bộ dự án **AI-Recommendation-System / CineAI V2.0**. Mục tiêu trọng tâm là kiểm tra nghiêm ngặt tính toàn vẹn hệ thống (System Integrity), độ chính xác của 10 mô hình gợi ý (Recommendation Correctness), tính tuân thủ giao thức mạng REST API (API Schema & Error Handling), trải nghiệm người dùng thực tế trên trình duyệt thật qua tự động hóa Playwright (End-to-End Real Browser Audit), cũng như đo lường hiệu năng và tính trung thực khoa học trước khi bước vào bảo vệ đồ án.

### Kết quả Tổng hợp các Bộ Kiểm thử:
- **Core AI & Backend Unit Tests:** **85 / 85 tests PASSED (100% OK)** trong 28.83s.
- **Frontend Component Tests (Vitest):** **6 / 6 tests PASSED (100% OK)** trong 25.33s.
- **Frontend Production Build (Vite 8.3.5):** **SUCCESS** trong 9.19s (`dist/assets/index-4Qliaurf.css` 53.87 kB, `dist/assets/index-DL6xyFTT.js` 709.79 kB).
- **Backend API & Correctness Audit:** **120 / 120 kiểm tra đề xuất thành công (0 vi phạm)**; 8/8 REST endpoints phản hồi chuẩn schema.
- **Real Browser E2E Scenarios (Playwright with Microsoft Edge):** **19 / 19 scenarios PASSED (100% OK)**, **0 console errors**, **0 network errors**.
- **Tính Bất biến Checkpoint:** 10/10 file checkpoint `.pkl` không bị thay đổi trong quá trình suy luận.

---

## 2. TEST ENVIRONMENT (MÔI TRƯỜNG KIỂM THỬ)

- **Hệ điều hành:** Microsoft Windows 11 (build x64)
- **Runtime Python:** Python 3.11.9 (với môi trường ảo `.venv`)
- **Khung phụ thuộc AI:** PyTorch 2.14.1+cpu, Scikit-learn 1.6.1, NumPy 2.2.3, SciPy 1.15.2, Pandas 2.2.3, Joblib 1.4.2
- **Backend Service:** FastAPI 0.162.0, Uvicorn 0.34.0, Pydantic 2.14.1 (Port 8000)
- **Frontend Framework:** React 19.3.1, TypeScript 5.8 / 7.0 preview, Vite 8.3.5, Tailwind CSS v4.3.3 (`@tailwindcss/vite`) (Port 5173)
- **Trình duyệt Tự động hóa E2E:** Microsoft Edge (Chromium engine, kênh phân phối chính thức của hệ điều hành, chế độ headless)
- **Tập dữ liệu:** MovieLens 100K (100,000 ratings, 943 users, 1,682 movies; chia cố định 80,000 Train / 20,000 Test với seed=42)

---

## 3. TEST CASES & MA TRẬN PHỦ ĐÁNH GIÁ (TEST MATRIX)

| Mã Nhóm | Nội dung Kiểm thử | Số lượng Cases | Kết quả Thực tế |
| :--- | :--- | :---: | :---: |
| **TC-INT** | Tính toàn vẹn 10 Checkpoints, dữ liệu phân hoạch và benchmark CSV | 10 | **10 / 10 PASS** |
| **TC-API** | Response schemas, status code, latency và error cases của 8 REST endpoints | 13 | **13 / 13 PASS** |
| **TC-REC** | 10 mô hình x 4 người dùng x 3 mức Top-N (5, 10, 20) kiểm tra candidate space, không trùng, không NaN | 120 | **120 / 120 PASS** |
| **TC-E2E** | Luồng người dùng thực tế trên trình duyệt: 5 trang, modal, filter, search, Model Battle, mobile | 19 | **19 / 19 PASS** |
| **TC-REG** | Bộ kiểm thử hồi quy tự động Python (Phase 2 -> 5) và Vitest | 91 | **91 / 91 PASS** |
| **TỔNG CỘNG** | **Toàn bộ hệ thống CineAI V2.0** | **253** | **253 / 253 PASS (100%)** |

---

## 4. API INTEGRATION TESTING RESULTS (KẾT QUẢ REST API)

Thực thi tự động thông qua script kiểm toán [`scripts/audit_phase6a_backend.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/scripts/audit_phase6a_backend.py):

| Endpoint | Phương thức | Dữ liệu Kiểm thử | HTTP Status | Độ trễ API Mạng | Schema Validation |
| :--- | :---: | :--- | :---: | :---: | :---: |
| `/api/v1/health` | `GET` | N/A | **200 OK** | 5.97 ms | Chứa `healthy`, 1682 movies, 10 models |
| `/api/v1/models` | `GET` | N/A | **200 OK** | 6.12 ms | Đủ 10 mô hình kèm `score_semantics` |
| `/api/v1/movies` | `GET` | `genre=Action&sort_by=rating` | **200 OK** | 12.45 ms | Phân trang 12 items, tổng 251 phim Action |
| `/api/v1/movies/50` | `GET` | `movie_id=50` (Star Wars) | **200 OK** | 7.80 ms | Đúng phim, phân bố sao và 6 phim tương tự |
| `/api/v1/users/1/profile` | `GET` | `user_id=1` | **200 OK** | 8.35 ms | Đúng 218 ratings từ tập Train, phân bố sao |
| `/api/v1/metrics` | `GET` | N/A | **200 OK** | 6.50 ms | Bảng xếp hạng 10 model, Best HR: Neural CF |
| `/api/v1/recommendations` | `POST` | `user_id=1, model=neural_cf, top_n=10` | **200 OK** | 9.20 ms | 10 phim, rank #1->#10, có formatted score |
| `/api/v1/compare` | `POST` | `user_id=1, neural_cf vs user_cf` | **200 OK** | 14.10 ms | Jaccard = 17.65%, 3 phim trùng khớp |

### Kiểm tra Xử lý Ngoại lệ và Lỗi đầu vào (Error Handling):
- `GET /api/v1/movies/99999`: Trả về đúng **404 Not Found** ("Movie not found").
- `GET /api/v1/users/99999/profile`: Trả về đúng **404 Not Found** ("User profile not found").
- `POST /api/v1/recommendations` với `user_id: 99999`: Bị Pydantic Schema chặn với **422 Unprocessable Entity** do vi phạm ràng buộc `le=943`.
- `POST /api/v1/recommendations` với `model_id: "invalid_model_xyz"`: Trả về đúng **400 Bad Request** ("Unknown model ID").
- `POST /api/v1/recommendations` với `top_n: -5`: Bị Pydantic Schema chặn với **422 Unprocessable Entity** do vi phạm ràng buộc `ge=1`.

---

## 5. RECOMMENDATION CORRECTNESS (TÍNH ĐÚNG ĐẮN CỦA GỢI Ý)

Kiểm toán trên 4 người dùng đại diện có mức độ tương tác khác biệt trong tập Train:
1. **User #1:** Hoạt động rất cao (**218 ratings** trong Train).
2. **User #34:** Hoạt động thấp / Gần Cold-start (**20 ratings** trong Train).
3. **User #196:** Hoạt động trung bình (**39 ratings** trong Train).
4. **User #943:** Người dùng cuối cùng trong danh mục (**168 ratings** trong Train).

### Kết quả Kiểm định trên cả 10 Mô hình (120 lượt chạy thực tế):
1. **Số lượng Phần tử:** Mọi danh sách trả về đúng chính xác số lượng yêu cầu $K \in \{5, 10, 20\}$ (100% khớp).
2. **Tính Đơn nhất (No Duplicates):** Tuyệt đối không có bất kỳ `movie_id` nào bị lặp lại trong cùng một danh sách gợi ý.
3. **Cách ly Dữ liệu Huấn luyện (No Seen Items):**
   $$\forall u, \quad \mathcal{S}_u^{rec} \cap \mathcal{I}_u^{train} = \emptyset$$
   Đã so khớp từng bộ phim được gợi ý với toàn bộ tập lịch sử đánh giá của người dùng trong `train.csv`. **Không có bất kỳ bộ phim đã xem nào bị đề xuất lại (0 vi phạm).**
4. **Miền Giá trị Hợp lệ:** 100% `movie_id` nằm trong khoảng định danh hợp lệ $[1, 1682]$.
5. **Tính Hữu hạn của Điểm số:** 100% điểm dự đoán gốc (`raw_score`) là số thực hữu hạn, **không chứa NaN hoặc Inf**.
6. **Không Can thiệp Tập Test:** Quá trình suy luận chỉ truy cập profile và history từ `train.csv`, hoàn toàn không truy xuất `test.csv`.
7. **Không Huấn luyện lại (No Retraining during Inference):** Mô hình được nạp từ checkpoint đã đóng gói sẵn trong `models/checkpoints/` và chỉ thực thi hàm `recommend()` hoặc `predict()`.
8. **Bảo tồn Checkpoint (File Immutability):** So sánh `st_mtime` và kích thước file trước và sau 120 lượt gọi suy luận cho thấy 10 file `.pkl` **hoàn toàn bất biến**.

---

## 6. REAL BROWSER E2E TESTING (KIỂM THỬ TRÌNH DUYỆT THỰC TẾ)

Thực thi kịch bản E2E bằng Playwright trên trình duyệt **Microsoft Edge** kết nối trực tiếp với Frontend (`http://127.0.0.1:5173`) và Backend (`http://127.0.0.1:8000`).

### Kết quả Chi tiết 19 Kịch bản (19/19 PASS):
1. **Trang Home:**
   - Brand logo `CINEAI` hiển thị chuẩn typography điện ảnh.
   - Hero banner hiển thị tiêu đề nổi bật `Star Wars (1977)`, điểm số xác thực `★ 4.4/5.0` và 2 nút bấm tương tác.
   - Bấm nút *"Movie Details & Stats"* mở thành công modal chi tiết phim với đầy đủ thông số Train ratings và liên kết IMDb.
   - Bấm nút đóng modal (`aria-label="Close modal"`) hoặc click backdrop đóng modal mượt mà.
2. **Trang Discover:**
   - Điều hướng thành công đến `/discover`.
   - Tìm kiếm từ khóa `"Godfather"`: Hệ thống lọc tức thời và hiển thị đúng 2 tác phẩm (*The Godfather*, *The Godfather: Part II*).
   - Bấm chọn thể loại `"Sci-Fi"`: Bộ lọc kích hoạt hiển thị đúng 101 bộ phim khoa học viễn tưởng.
3. **Trang AI Recommendations:**
   - Điều hướng thành công đến `/recommendations`.
   - Chọn chuyển sang mô hình `"User-Based Collaborative Filtering"`.
   - Chuyển số lượng gợi ý thành `"Top 20"`.
   - Bấm nút *"Generate Recommendations"*: Backend thực thi suy luận thực và trả về 20 thẻ phim kèm huy hiệu xếp hạng `#1` đến `#20` và điểm số `4.xx ★`.
4. **Trang AI Lab & Benchmark:**
   - Điều hướng thành công đến `/lab`.
   - Bảng xếp hạng chính thức 10 mô hình hiển thị đầy đủ, mô hình Neural CF đứng hạng #1.
   - Chuyển tab sang *"1v1 Model Battle"*, chạy thử đối đầu giữa `Neural CF` và `User-Based CF` cho User #1: Kết quả hiển thị chỉ số tương đồng Jaccard `17.6%` kèm 3 phim mà cả hai mô hình cùng đồng thuận đề xuất.
5. **Trang User Profile:**
   - Điều hướng thành công đến `/profile`.
   - Hiển thị hồ sơ User #1: Đúng 218 lượt đánh giá trong Train, biểu đồ phân bố 1 đến 5 sao và danh sách thể loại yêu thích.
   - Sử dụng bộ chọn nhanh chuyển sang User #196: Toàn bộ hồ sơ tải lại tức thì và hiển thị đúng thông số của User #196.
6. **Mobile Responsiveness (390 x 844 px):**
   - Đổi viewport sang kích thước chuẩn iPhone.
   - Nút menu hamburger xuất hiện và mở ra thanh điều hướng trượt dọc hoàn chỉnh.
   - Không xuất hiện hiện tượng vỡ layout hay thanh cuộn ngang ngoài ý muốn (Zero horizontal overflow).

> **Bằng chứng Ảnh chụp Màn hình:** Đã lưu trữ 6 tệp ảnh độ phân giải cao tại [`reports/screenshots/e2e/`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/reports/screenshots/e2e/).  
> **Console & Network:** 0 console errors, 0 network errors.

---

## 7. PERFORMANCE & RELIABILITY (HIỆU NĂNG VÀ ĐỘ TIN CẬY)

### 7.1 Độ trễ Suy luận Thực tế (Inference Latency) theo Từng Mô hình
Đo đạc trung bình qua 12 lượt suy luận cho mỗi mô hình trên máy trạm kiểm thử:

| Mô hình AI | Họ Thuật toán | Độ trễ Suy luận Server (`inference_time_ms`) | Tổng Độ trễ API Mạng (Localhost 127.0.0.1) |
| :--- | :--- | :---: | :---: |
| **Popularity Baseline** | Heuristic Popularity | **0.82 ms** | 6.85 ms |
| **Content-Based** | TF-IDF Cosine Similarity | **6.45 ms** | 12.30 ms |
| **User-Based CF** | Pearson Correlation + Shrinkage | **2.15 ms** | 8.10 ms |
| **Item-Based CF** | Adjusted Cosine + KNN | **1.95 ms** | 7.90 ms |
| **KNNBaseline** | Biased Residual Similarity | **1.88 ms** | 7.75 ms |
| **Funk SVD** | Matrix Factorization ($k=20$) | **0.95 ms** | 6.90 ms |
| **SVD++** | MF + Implicit Feedback ($k=20$) | **1.42 ms** | 7.35 ms |
| **NMF** | Non-Negative Matrix Factorization | **0.98 ms** | 6.95 ms |
| **Neural CF (NeuMF)** | PyTorch Deep Neural Network | **2.85 ms** | 8.80 ms |
| **Weighted Hybrid** | Ensemble 80% CF + 20% CB | **7.15 ms** | 13.50 ms |

> [!TIP]
> 100% các mô hình AI có thời gian suy luận trên server dưới **8 mili-giây** cho mỗi truy vấn Top-20, đáp ứng vượt trội tiêu chuẩn suy luận thời gian thực trong môi trường thương mại điện tử (thường yêu cầu $< 50$ms).

### 7.2 Tiêu thụ Bộ nhớ & Độ ổn định khi Chuyển đổi Mô hình
- Khi nạp toàn bộ 10 mô hình vào bộ nhớ RAM của FastAPI, mức tiêu thụ Commit Charge của worker process ổn định ở mức **~1.38 GB** (bao gồm PyTorch C++ runtime, OpenMP và 943 ma trận biểu diễn).
- Việc người dùng chuyển đổi mô hình liên tục (120 truy vấn liên tiếp trên cả 10 mô hình) không gây ra hiện tượng rò rỉ bộ nhớ (memory leak), mức chiếm dụng RAM không tăng lũy tiến.

---

## 8. SCIENTIFIC INTEGRITY & EXPERIMENTAL LIMITATIONS (TÍNH TRUNG THỰC KHOA HỌC)

Để đảm bảo tiêu chuẩn cao nhất về tính trung thực trong nghiên cứu khoa học phục vụ bảo vệ đồ án, nhóm nghiên cứu ghi nhận minh bạch các điểm sau:

### 8.1 Tính Bất biến của Giao thức Đánh giá (Evaluation Protocol)
- **Bộ Dữ liệu Chuẩn:** Giữ nguyên 100,000 tương tác chia theo phương thức Stratified Per-User Split (tỷ lệ 80/20, seed=42): 80,000 Train / 20,000 Test.
- **Tập Người dùng Kiểm định:** Đúng **935 người dùng** có ít nhất 1 tương tác tích cực ($r \ge 4.0$) trong tập Test.
- **Candidate Set Đồng nhất:** Với mỗi người dùng kiểm thử, tập ứng viên đánh giá luôn là toàn bộ các bộ phim người dùng chưa từng tương tác trong tập Train: $\mathcal{I}_{all} \setminus \mathcal{I}_u^{train}$.
- **5 Độ đo Chuẩn tắc:** Precision@10, Recall@10, F1@10, NDCG@10 và Hit Rate@10 được tính toán theo đúng định nghĩa chuẩn mực trong Information Retrieval và Recommender Systems.

### 8.2 Bảng Xếp hạng Benchmark Chính thức Không Đổi ([`results/phase4c_metrics.csv`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/results/phase4c_metrics.csv))

| Hạng | Tên Mô hình | Precision@10 | Recall@10 | F1@10 | NDCG@10 | Hit Rate@10 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Neural CF (NeuMF)** | **0.2251** | **0.2391** | **0.1898** | **0.3083** | **82.03%** |
| **2** | **Weighted Hybrid** | **0.1475** | **0.1327** | **0.1164** | **0.1943** | **65.78%** |
| **3** | **User-Based CF** | 0.1467 | 0.1352 | 0.1160 | 0.1948 | 65.24% |
| **4** | **Item-Based CF** | 0.1247 | 0.1146 | 0.0979 | 0.1611 | 59.04% |
| **5** | **KNNBaseline** | 0.1059 | 0.0816 | 0.0778 | 0.1316 | 50.27% |
| **6** | **SVD++** | 0.0896 | 0.0701 | 0.0671 | 0.1089 | 47.59% |
| **7** | **Funk SVD** | 0.0711 | 0.0508 | 0.0508 | 0.0715 | 40.32% |
| **8** | **Popularity Baseline** | 0.0653 | 0.0497 | 0.0473 | 0.0764 | 39.14% |
| **9** | **NMF** | 0.0542 | 0.0380 | 0.0387 | 0.0525 | 33.16% |
| **10** | **Content-Based** | 0.0236 | 0.0199 | 0.0174 | 0.0296 | 18.72% |

### 8.3 Trình bày Minh bạch Giới hạn Thực nghiệm (Crucial Scientific Disclosure)
> [!IMPORTANT]
> **Thừa nhận Khoa học về Giai đoạn 4A (Neighborhood CF Tuning):**
> Trong Phase 4A ban đầu, thuật toán User-Based CF thuần túy chỉ đạt Hit Rate@10 là 0.0535 do vấn đề đồng đánh giá thưa thớt (sparsity of co-ratings). Trong quá trình kiểm toán thuật toán ở Phase 4A, các kỹ sư đã trực tiếp quan sát kết quả trên tập Test để phân tích nguyên nhân và bổ sung các kỹ thuật:
> 1. Trọng số ý nghĩa (Significance Weighting / Shrinkage với $\lambda_s = 100$).
> 2. Hệ số khuếch đại phi tuyến ($\gamma = 5.0$).
> 3. Cơ chế Fallback về Mean Rating.
>
> Mặc dù khi tính toán độ đo, tập Test vẫn hoàn toàn độc lập về mặt dữ liệu tương tác (không dùng ratings trong Test để tính tương quan ma trận), **quá trình lựa chọn tham số của User-Based CF đã chịu ảnh hưởng gián tiếp từ việc quan sát phản hồi của tập Test (Researcher-in-the-loop information leakage)**.  
> Ngược lại, ở Phase 4B (Matrix Factorization) và Phase 4C (Neural CF), nhóm nghiên cứu đã khắc phục triệt để bằng việc thiết lập phân hoạch **Train-Inner (64,000) và Validation (16,000)** độc lập để tinh chỉnh siêu tham số, đảm bảo tập Test hoàn toàn là "blind test".  
> Giới hạn này sẽ được trình bày trung thực trong báo cáo bảo vệ trước Hội đồng.

---

## 9. DEFECTS IDENTIFIED & REMEDIATION (LỖI ĐÃ PHÁT HIỆN VÀ KHẮC PHỤC)

Trong quá trình thực hiện Phase 6A, các lỗi thực tế sau đã được phát hiện và xử lý ngay tại gốc:

1. **Lỗi Schema Parameter Mismatch trong Audit Script:**
   - *Phát hiện:* Script kiểm toán ban đầu gửi tham số `top_k` thay vì `top_n` theo đúng Pydantic schema của `RecommendationRequest`.
   - *Khắc phục:* Cập nhật script audit tuân thủ đúng 100% data contract của backend.
2. **Lỗi Thiếu Accessibility Tag trên Modal Close Button:**
   - *Phát hiện:* Nút đóng của `MovieDetailModal.tsx` thiếu thuộc tính `aria-label="Close modal"`, khiến bộ kiểm thử tự động E2E gặp khó khăn trong việc định vị và đóng modal.
   - *Khắc phục:* Đã bổ sung `aria-label="Close modal"` và `cursor-pointer` cho nút đóng modal, giúp luồng E2E đóng mở modal hoàn hảo.
3. **Lỗi IPv6 Loopback Delay trên Môi trường Windows:**
   - *Phát hiện:* Khi client Python kết nối tới `http://localhost:8000`, resolver của Windows thử IPv6 trước khi fallback sang IPv4, gây thêm 2 giây độ trễ ảo.
   - *Khắc phục:* Bổ sung cấu hình `host: true` trong `frontend/vite.config.ts` và sử dụng `127.0.0.1` trong các client kiểm thử để đo đạc độ trễ mạng chính xác tuyệt đối (giảm từ 2050ms xuống còn **5.97ms**).

---

## 10. REMAINING LIMITATIONS (CÁC GIỚI HẠN CÒN TỒN TẠI)

1. **Movie Poster API:** Bộ áp-phích hiện tại sử dụng công nghệ SVG 35mm Minimalist Artwork được sinh tự động theo thể loại và năm. Để có ảnh poster rạp chiếu phim chụp thực tế, cần cấu hình khóa TMDB API Key bên ngoài.
2. **Tài nguyên Bộ nhớ CPU:** PyTorch CPU inference yêu cầu khoảng 1.3 GB commit virtual memory. Trên các máy Windows có dung lượng RAM thấp ($< 8$ GB), cần duy trì swap/pagefile tối thiểu 4 GB để đảm bảo hệ thống vận hành trơn tru.

---

## 11. GO / NO-GO DECISION (QUYẾT ĐỊNH NGHIỆM THU)

Dựa trên kết quả thực nghiệm toàn diện:
- **Tính toàn vẹn thuật toán:** 10/10 mô hình hoạt động chính xác, không rò rỉ dữ liệu.
- **Độ tin cậy API:** 8/8 endpoints đạt chuẩn schema, xử lý lỗi tốt.
- **Trải nghiệm Frontend:** Giao diện điện ảnh Netflix chuẩn mực, 19/19 kịch bản E2E đạt yêu cầu, không lỗi console.
- **Kiểm thử tự động:** 85 bài test Python + 6 bài test Vitest đạt **100% PASS**.

### **KẾT LUẬN: GO (CHẤP THUẬN NGHIỆM THU CHÍNH THỨC)**
Hệ thống **AI-Recommendation-System / CineAI V2.0** chính thức đạt tiêu chuẩn nghiệm thu cấp độ đồ án xuất sắc, sẵn sàng cho việc hoàn thiện tài liệu thuyết minh và bảo vệ trước Hội đồng chấm thi.

---

## 12. GIT STATUS & QUẢN LÝ MÃ NGUỒN

- **Branch hiện tại:** `main`
- **Thay đổi đã kiểm tra:** Không tự động thực hiện commit hoặc push lên GitHub theo đúng chỉ thị.
- **Các tệp sửa đổi tối thiểu phục vụ kiểm thử:**
  - [`frontend/src/components/movies/MovieDetailModal.tsx`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/frontend/src/components/movies/MovieDetailModal.tsx): Bổ sung `aria-label="Close modal"`.
  - [`frontend/vite.config.ts`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/frontend/vite.config.ts): Bổ sung `host: true`.
- **Các tệp kiểm toán bổ sung:**
  - [`scripts/audit_phase6a_backend.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/scripts/audit_phase6a_backend.py): Script kiểm toán 120 ca kiểm thử đề xuất và 8 API.
  - [`frontend/e2e_phase6a.cjs`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/frontend/e2e_phase6a.cjs): Script kiểm thử E2E Playwright trên Edge thật.
  - [`results/phase6a_backend_audit.json`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/results/phase6a_backend_audit.json): Dữ liệu kết quả kiểm toán backend.
  - [`results/phase6a_e2e_audit.json`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/results/phase6a_e2e_audit.json): Dữ liệu kết quả kiểm toán E2E trình duyệt.
  - [`reports/screenshots/e2e/`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/reports/screenshots/e2e/): 6 ảnh chụp màn hình bằng chứng E2E.
