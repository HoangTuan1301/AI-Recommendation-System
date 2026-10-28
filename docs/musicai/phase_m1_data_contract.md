# HỢP ĐỒNG DỮ LIỆU MUSICAI (DATA CONTRACT FOR PHASE M2)
## PHASE M1 SPECIFICATION: DATA SCHEMAS, SPLIT PROTOCOLS & MODEL ACCESS INTERFACE

**Dự án:** Multi-Domain AI Recommendation System (MusicAI)  
**Tài liệu:** Hợp đồng Dữ liệu & Giao diện truy xuất cho Phase M2  
**Trạng thái:** CHUẨN HÓA & KHÓA GIAO DIỆN (LOCKED SPECIFICATION)  
**Tác giả:** Senior Data Engineer & Machine Learning Quality Engineer  
**Ngày:** 28/10/2026  

---

## 1. MỤC ĐÍCH VÀ PHẠM VI ÁP DỤNG

Tài liệu này xác lập hợp đồng dữ liệu chính thức giữa **Giai đoạn M1 (Kỹ thuật Dữ liệu)** và **Giai đoạn M2 (Huấn luyện Mô hình AI)**.
Mọi mô hình học máy (iALS, BPR-MF, Implicit NeuMF, Item-CF, User-CF, Popularity) trong Giai đoạn M2 bắt buộc phải tiêu thụ dữ liệu theo đúng cấu trúc, kiểu dữ liệu và ràng buộc được quy định tại đây, tuyệt đối không được tự ý sinh trường dữ liệu giả lập.

---

## 2. RÀNG BUỘC CỐT LÕI: PHẢN HỒI NGẦM ĐỊNH (IMPLICIT FEEDBACK)

1. **Tuyệt đối không coi `play_count` là điểm số 1–5 sao:**
   - Số lượt nghe (`play_count`) là biến số đếm tần suất hành vi không bị chặn trên ($r_{ui} \in \mathbb{N}^+$), mang tính chất ngầm định.
   - Không được ép kiểu hay chuẩn hóa Min-Max tuyến tính `play_count` về thang đo 1-5 sao như phim ảnh.
2. **Quy tắc chuyển đổi sở thích nhị phân (Binary Preference $p_{ui}$):**
   $$p_{ui} = \begin{cases} 1 & \text{nếu } r_{ui} \ge 1 \\ 0 & \text{nếu chưa quan sát (unobserved)} \end{cases}$$
3. **Quy tắc chuyển đổi độ tin cậy ngầm định (Confidence Level $c_{ui}$):**
   $$c_{ui} = 1 + \alpha \cdot \log(1 + r_{ui}) \quad \text{với } \alpha = 40.0$$
4. **Định nghĩa tương tác tích cực (Positive Ground Truth):**
   Mọi bản ghi xuất hiện trong tập kiểm thử `test.csv` đều là tương tác tích cực thực tế ($p_{ui} = 1$). Không áp dụng ngưỡng $r \ge 4.0$ của CineAI.

---

## 3. ĐỊNH NGHĨA CÁC ĐẶC TẢ LƯU TRỮ (DATA SCHEMAS)

Toàn bộ các tệp dữ liệu được lưu trữ tại `data/music/processed/` dưới định dạng CSV (UTF-8, ngăn cách bằng dấu phẩy):

### 3.1. Tệp tương tác người dùng - bài hát (`interactions.csv`)
Tập hợp toàn bộ tương tác sau khi lọc $k$-core:
- `user_id` (string, required): Mã định danh duy nhất của người dùng từ Echo Nest (ví dụ: `b80344d063b5ccb3212f76538f3d9e43d87dca9e`).
- `song_id` (string, required): Mã bài hát duy nhất từ Million Song Dataset (ví dụ: `SOAKIMP12A8C130995`).
- `play_count` (int64, required): Số lần nghe hợp lệ ($r_{ui} \ge 1$).

### 3.2. Tập phân hoạch (`train.csv`, `validation.csv`, `test.csv`)
Cùng kế thừa định dạng chuẩn:
- `user_id` (string): Khóa ngoại tham chiếu người dùng.
- `song_id` (string): Khóa ngoại tham chiếu bài hát.
- `play_count` (int64): Tần suất nghe trong phân hoạch tương ứng.

### 3.3. Tệp danh mục bài hát và metadata (`songs.csv`)
- `song_id` (string, Primary Key): Mã định danh chuẩn MSD (18 ký tự).
- `title` (string): Tiêu đề bài hát nguyên bản (UTF-8).
- `artist_name` (string): Tên nghệ sĩ hoặc nhóm nhạc biểu diễn.
- `track_id` (string): Mã nhận diện bản ghi âm thanh 7digital (ví dụ: `TRMMMYQ128F932D901`).

### 3.4. Bảng ánh xạ chỉ số nguyên (`user_mapping.csv`, `song_mapping.csv`)
Phục vụ tính toán ma trận thưa và tầng Embedding của mạng nơ-ron:
- `user_mapping.csv`:
  - `raw_user_id` (string): Mã chuỗi hash gốc.
  - `user_idx` (int32): Chỉ số nguyên liên tục từ $0$ đến $|U|-1$.
- `song_mapping.csv`:
  - `raw_song_id` (string): Mã chuỗi gốc.
  - `song_idx` (int32): Chỉ số nguyên liên tục từ $0$ đến $|I|-1$.

---

## 4. QUY TẮC RÀNG BUỘC PHÂN HOẠCH (SPLITTING CONSTRAINTS)

Để bảo đảm tính khách quan khoa học và tránh rò rỉ dữ liệu (data leakage):
1. **Phân hoạch phân tầng theo người dùng (User-Aware Holdout):**
   - Với mỗi người dùng, tương tác được phân chia ngẫu nhiên cố định (`seed=42`) theo tỷ lệ:
     - Tập Huấn luyện (**Train**): ~80% tương tác.
     - Tập Thẩm định (**Validation**): ~10% tương tác.
     - Tập Kiểm thử (**Test**): ~10% tương tác.
2. **Tính rời rạc tuyệt đối (Mutual Exclusivity):**
   $$(\text{Train} \cap \text{Validation}) = \emptyset, \quad (\text{Train} \cap \text{Test}) = \emptyset, \quad (\text{Validation} \cap \text{Test}) = \emptyset$$
3. **Bảo đảm không gian danh mục (Warm-Start Catalog Coverage):**
   Mọi bài hát trong `validation.csv` và `test.csv` bắt buộc phải xuất hiện ít nhất 1 lần trong `train.csv`. Các bài hát không có trong tập Train sẽ được xử lý loại trừ trong bước phân hoạch để bảo đảm bài toán Collaborative Filtering có thể đánh giá được.
4. **Bảo tồn tập Test:**
   Tập `test.csv` tuyệt đối không được sử dụng để tinh chỉnh siêu tham số (hyperparameter tuning). Tập `validation.csv` là tập duy nhất được dùng cho việc chọn mô hình trong Phase M2.

---

## 5. GIAO DIỆN TRUY XUẤT DỮ LIỆU CHO PHASE M2 (ACCESS INTERFACE)

Các mô hình trong Phase M2 sẽ truy xuất dữ liệu thông qua lớp nạp `MusicDatasetLoader` chuẩn:

```python
class MusicDatasetLoader:
    """Truy xuất tập dữ liệu MusicAI đã tiền xử lý phục vụ huấn luyện Phase M2."""
    
    def get_train_df(self) -> pd.DataFrame:
        """Trả về DataFrame tương tác tập Train [user_id, song_id, play_count]."""
        ...
        
    def get_val_df(self) -> pd.DataFrame:
        """Trả về DataFrame thẩm định [user_id, song_id, play_count]."""
        ...
        
    def get_test_df(self) -> pd.DataFrame:
        """Trả về DataFrame kiểm thử [user_id, song_id, play_count]."""
        ...
        
    def get_song_catalog(self) -> pd.DataFrame:
        """Trả về DataFrame metadata bài hát [song_id, title, artist_name, track_id]."""
        ...
        
    def get_sparse_interaction_matrix(self) -> scipy.sparse.csr_matrix:
        """Trả về ma trận thưa csr_matrix kích thước (|U|, |I|) chứa giá trị confidence c_ui."""
        ...
```

---

## 6. KẾT LUẬN

Hợp đồng dữ liệu Phase M1 đã hoàn thiện, bảo đảm sự tách bạch tuyệt đối giữa việc thu thập dữ liệu và thuật toán gợi ý, loại trừ 100% rủi ro phụ thuộc vào điểm số sao tường minh của CineAI.
