# GIAO THỨC HUẤN LUYỆN VÀ THẨM ĐỊNH MÔ HÌNH MUSICAI
## PHASE M2: TRAINING PROTOCOL & VALIDATION BENCHMARK SPECIFICATION

**Dự án:** Multi-Domain AI Recommendation System (CineAI V2.0 & MusicAI)  
**Nhánh Git:** `feature/musicai-models`  
**Tài liệu:** Giao thức Huấn luyện Tuần tự & Thẩm định Khoa học  
**Trạng thái:** HOÀN THÀNH HUẤN LUYỆN & XÁC THỰC THỰC NGHIỆM  
**Ngày:** 28/10/2026  

---

## 1. NGUYÊN TẮC HUẤN LUYỆN TUẦN TỰ (SEQUENTIAL TRAINING DISCIPLINE)

Nhằm bảo đảm tài nguyên máy trạm (laptop cá nhân) không bị quá tải CPU hay tràn RAM, hệ thống áp dụng chiến lược huấn luyện tuần tự từng mô hình độc lập qua kịch bản [`src/music/training/train_all_models.py`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/src/music/training/train_all_models.py):

1. **Thứ tự thực thi:**
   - Bước 1: `PopularityMusicRecommender` (Khởi tạo mốc cơ sở)
   - Bước 2: `ItemCFMusicRecommender` (Xây dựng đồ thị tương đồng bài hát)
   - Bước 3: `UserCFMusicRecommender` (Xây dựng đồ thị tương đồng người dùng)
   - Bước 4: `IALSMusicRecommender` (Phân rã ma trận nghiệm đóng ALS)
   - Bước 5: `BPRMFMusicRecommender` (Tối ưu hóa xếp hạng cặp SGD)
   - Bước 6: `ImplicitNeuralCFRecommender` (Huấn luyện mạng nơ-ron sâu NeuMF trên PyTorch)
2. **Quy trình kiểm tra tại từng bước (Post-Training Verification):**
   - Lưu checkpoint ngay sau khi khớp mô hình.
   - Thử nghiệm tải lại (Reload Test) từ đĩa và sinh Top-5 gợi ý cho người dùng mẫu.
   - Đo lường dung lượng RAM và thời gian chạy.
   - Chạy đánh giá thẩm định độc lập trên cùng một đối tượng `MusicTopNEvaluator`.

---

## 2. GIAO THỨC THẨM ĐỊNH (VALIDATION EVALUATION PROTOCOL)

1. **Tập dữ liệu thẩm định:** `data/music/processed/validation.csv` gồm **4,844 tương tác** thực tế từ 2,520 người dùng.
2. **Tập ứng viên dự đoán (Candidate Song Pool):**
   - Không dùng tập ứng viên rút gọn hay lấy mẫu giả.
   - Toàn bộ danh mục **2,706 bài hát** đều tham gia xếp hạng, sau khi đã loại trừ các bài hát người dùng đã nghe trong tập Train (`mask seen items`).
3. **5 Độ đo Top-10 chuẩn mực:**
   - **Hit Rate@10 (HR@10):** Tỷ lệ người dùng có ít nhất 1 bài hát trong danh sách gợi ý trúng đích tập Validation.
   - **Precision@10:** Tỷ lệ bài hát gợi ý trúng đích trên tổng số 10 bài gợi ý.
   - **Recall@10:** Tỷ lệ bài hát gợi ý trúng đích trên tổng số bài hát người dùng thực sự nghe trong tập Validation.
   - **NDCG@10:** Đo lường chất lượng xếp hạng có chiết khấu vị trí.
   - **MRR@10 (Mean Reciprocal Rank):** Nghịch đảo vị trí bài hát đúng đầu tiên trong Top-10.
4. **Nguyên tắc cô lập tập Test:** Tập `test.csv` được niêm phong 100%, không tham gia vào bất kỳ công đoạn nào của Phase M2.

---

## 3. THÔNG SỐ VÀ CẤU HÌNH SIÊU THAM SỐ ĐÃ HUẤN LUYỆN

| Mô hình AI | Siêu tham số chính (Hyperparameters) | Môi trường tính toán |
|---|---|---|
| **Popularity** | `weight = log(1 + listeners) * log(1 + plays)` | CPU (NumPy/Pandas) |
| **Item-Based CF** | $K=50$ neighbors, Cosine Similarity, $\alpha=40.0$, $\text{threshold}=0.01$ | CPU (SciPy CSR) |
| **User-Based CF** | $K=50$ neighbors, Cosine Similarity, $\alpha=40.0$, $\text{threshold}=0.01$ | CPU (SciPy CSR) |
| **Implicit ALS** | $k=32$ factors, $\lambda=0.1$, $\alpha=40.0$, 15 iterations, `seed=42` | CPU (NumPy Cholesky) |
| **BPR-MF** | $k=32$ factors, $\text{lr}=0.05$, $\text{reg}=0.01$, 20 epochs, batch=1024, `seed=42` | CPU (NumPy Mini-batch SGD) |
| **Implicit NeuMF** | GMF=32, MLP=32, Layers=[64, 32, 16], Dropout=0.2, lr=0.001, neg_ratio=4, 12 epochs | CPU (PyTorch DataLoader) |

---

## 4. QUẢN LÝ CHECKPOINT VÀ NGUYÊN TẮC AN TOÀN BỘ NHỚ

- Toàn bộ 6 checkpoint được lưu trữ biệt lập tại: `models/checkpoints/music/*.pkl`.
- Tuyệt đối không can thiệp hay ghi đè vào 10 checkpoint điện ảnh trong `models/checkpoints/*.pkl`.
- Đã được cấu hình trong `.gitignore` ngăn ngừa commit tệp nhị phân lớn lên GitHub.
