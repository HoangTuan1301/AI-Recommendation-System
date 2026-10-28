# NGHIÊN CỨU KHẢ THI VÀ LỰA CHỌN TẬP DỮ LIỆU ÂM NHẠC
## PHASE M0: MUSIC DATASET SELECTION & FEASIBILITY STUDY

**Dự án:** Multi-Domain AI Recommendation System (MusicAI)  
**Tài liệu:** Đánh giá tính khả thi và Chiến lược dữ liệu âm nhạc  
**Trạng thái:** HOÀN TẤT & ĐỀ XUẤT (GO FOR LAST.FM 2K HETREC 2011)  
**Tác giả:** Senior AI/ML Engineer & Recommender Systems Researcher  
**Ngày:** 28/10/2026  

---

## 1. MỤC TIÊU VÀ TIÊU CHÍ ĐÁNH GIÁ TẬP DỮ LIỆU

Trong khi miền phim ảnh (CineAI) sử dụng tập dữ liệu kinh điển MovieLens 100K với điểm đánh giá tường minh 1-5 sao, miền âm nhạc (MusicAI) đòi hỏi một tập dữ liệu mang bản chất **phản hồi ngầm định (Implicit Feedback)**:
- Người dùng không chấm điểm từng bài hát mà tiêu thụ thông qua hành vi nghe nhạc (số lượt nghe - play count, số lần lặp lại - repeat count, lưu vào danh sách yêu thích).
- Yêu cầu hệ thống phải huấn luyện được trên máy trạm/laptop cá nhân (RAM 8GB - 16GB, CPU đa nhân) mà không gây tràn bộ nhớ (Out-Of-Memory).
- Tuân thủ chuẩn mực học thuật quốc tế (được trích dẫn rộng rãi tại ACM RecSys, KDD, SIGIR).
- Giữ vững tính đạo đức nghiên cứu: Không chứa hoặc phân phối tệp âm thanh bản quyền thương mại; chỉ sử dụng metadata, hành vi tương tác và định danh bài hát/nghệ sĩ.

---

## 2. NGHIÊN CỨU SO SÁNH 4 BỘ DỮ LIỆU ÂM NHẠC HÀNG ĐẦU

Chúng tôi tiến hành khảo sát và đo lường định lượng trên 4 tập dữ liệu âm nhạc công khai phổ biến nhất trong giới nghiên cứu Recommender Systems:

| Tiêu chí so sánh | Last.fm 2K (HetRec 2011) | Last.fm 1K (Celma 2010) | MSD Taste Profile (Echo Nest) | Spotify MPD (RecSys 2018) |
|---|---|---|---|---|
| **Số lượng người dùng ($|U|$)** | 1,892 người dùng | 992 người dùng | 1,019,318 người dùng | 1,000,000 playlist |
| **Số lượng mục ($|I|$)** | 17,632 nghệ sĩ | 176,948 nghệ sĩ / bài hát | 384,546 bài hát | 2,262,292 bài hát |
| **Số lượng tương tác ($|R|$)** | 92,834 cặp tương tác | ~19,150,868 sự kiện nghe | 48,373,586 lượt nghe | 66,346,428 bài trong playlist |
| **Bản chất phản hồi** | Ngầm định: Lượt nghe tích lũy (`weight` = listen count) | Ngầm định: Nhật ký sự kiện thời gian thực (User, Track, Timestamp) | Ngầm định: Cặp (User, Song, Play count) | Ngầm định: Đồng xuất hiện trong Playlist (Co-occurrence) |
| **Độ thưa thớt (Sparsity)** | 99.72% | > 99.9% | 99.98% | > 99.99% |
| **Dung lượng nén / giải nén** | **~15 MB / ~45 MB** | ~540 MB / ~2.6 GB | ~500 MB / ~3.0 GB | ~31 GB / ~120 GB |
| **Metadata kèm theo** | Tên nghệ sĩ, Tags/Genres, Mạng bạn bè (`user_friends`) | Tên nghệ sĩ, tên bài hát, mã MusicBrainz ID | ID bài hát, ID nghệ sĩ (cần join với Million Song) | Tên playlist, mô tả, số followers |
| **Tải bộ nhớ RAM (Training)** | **Rất nhẹ (< 500 MB RAM)** | Trung bình (2 - 4 GB RAM) | Rất nặng (> 12 GB RAM nếu không lọc) | Cực nặng (> 32 GB RAM) |
| **Thời gian huấn luyện 1 Epoch** | **< 2 giây (CPU)** | ~45 giây (CPU) | Hàng chục phút | Vài giờ |
| **Giấy phép & Bản quyền** | Nghiên cứu học thuật (Free Academic Use) | Giấy phép học thuật Celma / Last.fm API | Học thuật (Echo Nest / LabROSA) | Nghiên cứu học thuật có giới hạn |

---

## 3. PHÂN TÍCH ƯU - NHƯỢC ĐIỂM CHI TIẾT

### 3.1. Ứng viên 1: Last.fm 2K (HetRec 2011 - Celma et al.)
- **Ưu điểm vượt trội:**
  - Kích thước tập dữ liệu hoàn hảo cho môi trường máy trạm cá nhân: 1,892 người dùng, 17,632 nghệ sĩ, 92,834 tương tác.
  - Cung cấp sẵn lượt nghe tích lũy (`weight` = tổng số lần người dùng nghe nghệ sĩ đó), phản ánh chính xác hàm mức độ tin cậy $c_{ui} = 1 + \alpha r_{ui}$ trong mô hình **Implicit ALS (iALS)** và **BPR-MF**.
  - Đầy đủ thông tin thẻ gắn (`user_taggedartists.dat`, `tags.dat`) với hơn 186,000 lượt gán nhãn thể loại (rock, pop, indie, electronic, metal...), hỗ trợ xây dựng mô hình **Content-Based Music Recommender** chất lượng cao mà không cần crawl thêm dữ liệu ngoài.
  - Cấu trúc tệp dữ liệu dạng bảng phân tách bằng ký tự tab (`.dat`), định dạng sạch sẽ, không có dòng rác.
- **Hạn chế:**
  - Tương tác ở mức độ người dùng - nghệ sĩ (User-Artist), chưa chi tiết đến từng bản ghi bài hát (User-Track). Tuy nhiên, trong lý thuyết RecSys âm nhạc, bài toán User-Artist Recommendation là bài toán nền tảng, có ý nghĩa ứng dụng thực tiễn cao nhất và tránh hiện tượng quá thưa thớt (Extreme Cold-Start).

### 3.2. Ứng viên 2: Last.fm 1K (Celma 2010)
- **Ưu điểm:** Có thông tin mức bài hát kèm timestamp từng lượt nghe.
- **Hạn chế:** Tệp thô nặng hơn 2.5 GB, chứa hơn 19 triệu sự kiện; đòi hỏi bước tiền xử lý lọc ngưỡng khắt khe (k-core filtering), thời gian parse file lâu và dễ gây nghẽn I/O trên laptop Windows.

### 3.3. Ứng viên 3: Million Song Dataset (Taste Profile Subset)
- **Ưu điểm:** Quy mô bài hát lớn, là chuẩn benchmark uy tín của giới nghiên cứu học thuật.
- **Hạn chế:** Quy mô 48 triệu tương tác khiến việc huấn luyện mô hình học sâu và ma trận factorize trên máy trạm thông thường gặp rủi ro tràn RAM hoặc huấn luyện kéo dài nhiều giờ; phải tạo một bản trích mẫu (subsample) 2,000 người dùng $\times$ 5,000 bài hát, làm mất tính chuẩn hóa nguyên bản.

### 3.4. Ứng viên 4: Spotify Million Playlist Dataset (RecSys 2018)
- **Ưu điểm:** Dữ liệu thực tế khổng lồ từ người dùng Spotify.
- **Hạn chế:** Dung lượng hơn 30 GB nén, giải nén ra 120 GB JSON; cấu trúc định hướng bài toán hoàn thiện danh sách phát (Playlist Continuation) thay vì gợi ý cá nhân hóa đa mô hình Top-N. Hoàn toàn không khả thi để chạy trên môi trường phát triển hiện tại.

---

## 4. QUYẾT ĐỊNH LỰA CHỌN TẬP DỮ LIỆU CHÍNH THỨC

### 4.1. Lựa chọn số 1 (Primary Recommendation): **Last.fm 2K (HetRec 2011)**
Chúng tôi chính thức đề xuất **Last.fm 2K (HetRec 2011)** làm tập dữ liệu chuẩn cho **MusicAI**:
1. **Tính tương thích khoa học:** Tỷ lệ người dùng/mục ($1,892 / 17,632$) tương đương với MovieLens 100K ($943 / 1,682$), tạo sự đối xứng hoàn hảo trong việc so sánh hiệu năng kiến trúc đa miền.
2. **Hiệu suất tính toán:** Thời gian nạp dữ liệu < 0.5 giây; thời gian huấn luyện toàn bộ chuỗi mô hình < 2 phút; hoàn toàn bảo đảm sự mượt mà và phản hồi tức thì cho backend FastAPI và giao diện người dùng.
3. **Tính phong phú của Metadata:** Cung cấp sẵn hệ sinh thái tag âm nhạc phong phú giúp triển khai đầy đủ cả mô hình lọc cộng tác (CF), phân rã ma trận ngầm định (iALS, BPR) lẫn mô hình dựa trên nội dung (Tag/Genre TF-IDF).

### 4.2. Lựa chọn dự phòng (Secondary / Fallback): **MSD Taste Profile Sampled Subset**
Nếu đồ án yêu cầu cụ thể cấp độ bài hát (Track-level), hệ thống sẽ kích hoạt kịch bản dự phòng:
- Trích xuất tập con cân bằng từ MSD Taste Profile: 2,000 người dùng có tương tác nhiều nhất và 5,000 bài hát phổ biến nhất.
- Giữ dung lượng trong ngưỡng an toàn < 50 MB, bảo đảm tính ổn định hệ thống.

---

## 5. THIẾT KẾ ĐƯỜNG ỐNG TIỀN XỬ LÝ VÀ PHÂN HOẠCH DỮ LIỆU (DATA INGESTION SPEC)

### 5.1. Định dạng dữ liệu và biến đổi phản hồi ngầm định

Tập tin gốc `user_artists.dat` gồm 3 cột:
$$\text{Row} = (\text{userID}, \text{artistID}, \text{weight})$$

Trong đó `weight` biểu diễn số lượt nghe của người dùng đối với nghệ sĩ ($r_{ui} \ge 1$).

Áp dụng phương pháp biến đổi chuẩn hóa của Hu, Koren & Volinsky (IEEE ICDM 2008):
- **Sở thích nhị phân (Binary Preference):**
  $$p_{ui} = \begin{cases} 1 & \text{nếu } r_{ui} > 0 \\ 0 & \text{nếu } r_{ui} = 0 \end{cases}$$
- **Mức độ tin cậy ngầm định (Confidence Level):**
  $$c_{ui} = 1 + \alpha \cdot \log(1 + r_{ui}) \quad \text{với } \alpha = 40$$
  *(Hàm logarit giúp kiểm soát các giá trị ngoại lai - heavy listeners nghe hàng ngàn lần mà không làm bão hòa trọng số).*

### 5.2. Chiến lược phân hoạch (Data Split Strategy)

Để tránh hiện tượng rò rỉ dữ liệu (data leakage) và phản ánh đúng bài toán gợi ý thực tế:
- **Phân hoạch theo người dùng (User-Stratified Split):**
  - Tập Huấn luyện (**Train**): 80% tương tác của mỗi người dùng.
  - Tập Thẩm định (**Validation**): 10% tương tác của mỗi người dùng (dùng để dừng sớm Early Stopping và tinh chỉnh siêu tham số).
  - Tập Kiểm thử (**Test**): 10% tương tác còn lại của mỗi người dùng (dùng để xuất báo cáo benchmark khách quan).
- **Cố định ngẫu nhiên:** Sử dụng `seed = 42` bảo đảm 100% khả năng tái lập (Reproducibility).
- **Lưu trữ độc lập:**
  - `data/music/raw/` (lưu trữ tệp gốc Last.fm).
  - `data/music/processed/train.csv`, `val.csv`, `test.csv`, `artists_clean.csv`, `tags_clean.csv`.

---

## 6. GIẢI PHÁP VỀ BẢN QUYỀN VÀ TRẢI NGHIỆM TRÌNH PHÁT NHẠC (AUDIO PLAYER)

Tuân thủ nghiêm ngặt quy định không lưu trữ hay phát tán tệp âm thanh có bản quyền thương mại:
1. **Dữ liệu hiển thị (Metadata Display):** Sử dụng tên nghệ sĩ, danh sách bài hát tiêu biểu, thể loại và thẻ phong cách lấy từ kho dữ liệu mở HetRec / MusicBrainz.
2. **Ảnh đại diện & Ảnh bìa (Artist Cover Art):** Sử dụng hệ thống tạo bìa đĩa động đa sắc màu (Dynamic Vinyl & Waveform Art Generator) tương tự cơ chế SVG Poster Service thành công của CineAI.
3. **Mô phỏng trình phát nhạc (Simulated Audio Player):**
   - Tích hợp một thanh điều khiển phát nhạc trực quan (Playback Bar) với đầy đủ thanh tiến trình thời gian thực, nút Play/Pause, âm lượng và biểu đồ sóng âm (Waveform Visualizer) chuyển động nhịp nhàng.
   - Khi bấm Play, hệ thống có thể phát đoạn âm thanh chime chào mừng không bản quyền (Royalty-free) ngắn hoặc chạy trình đếm thời gian thực mô phỏng 30 giây nghe thử (30-second preview simulation).
   - Tuyệt đối không gọi hay stream tệp MP3 không được cấp phép.

---

## 7. KẾT LUẬN

Tập dữ liệu **Last.fm 2K (HetRec 2011)** đáp ứng xuất sắc mọi tiêu chuẩn khắt khe về mặt học thuật, tài nguyên máy tính, bản quyền và tính tương thích với kiến trúc mở rộng của dự án. Đề xuất sẵn sàng để triển khai vào Giai đoạn M1 ngay sau khi nhận được phê duyệt.
