# QUYẾT ĐỊNH KIẾN TRÚC: CHUYỂN DỊCH SANG GỢI Ý MỨC BÀI HÁT (SONG-LEVEL)
## PHASE M1 ARCHITECTURAL AMENDMENT: USER-SONG COLLABORATIVE FILTERING

**Dự án:** Multi-Domain AI Recommendation System (CineAI & MusicAI)  
**Tài liệu:** Tu chính án Kiến trúc & Quyết định lựa chọn thực thể gợi ý  
**Trạng thái:** CHÍNH THỨC BAN HÀNH (SUPERSEDES LAST.FM 2K FOR ENTITY LEVEL)  
**Nhánh Git:** `feature/musicai-foundation`  
**Tác giả:** Senior Data Scientist & Recommender Systems Researcher  
**Ngày:** 28/10/2026  

---

## 1. BỐI CẢNH VÀ QUYẾT ĐỊNH CỐT LÕI

Trong giai đoạn **Phase M0**, kế hoạch ban đầu đề xuất sử dụng tập dữ liệu **Last.fm 2K (HetRec 2011)** do kích thước nhỏ gọn (~45 MB) và tính ổn định cao trên laptop. Tuy nhiên, tập dữ liệu HetRec 2011 chỉ chứa tương tác ở mức **Người dùng – Nghệ sĩ (User–Artist Interactions)** thông qua tệp `user_artists.dat`.

Theo yêu cầu nghiệp vụ bắt buộc:
> **LỰA CHỌN B: Gợi ý từng bài hát cụ thể (Individual Songs/Tracks) ngay từ đầu, TUYỆT ĐỐI KHÔNG gợi ý nghệ sĩ thay cho bài hát.**

Quyết định này phản ánh chính xác bài toán thực tế của các nền tảng phát nhạc trực tuyến hàng đầu thế giới (Spotify, Apple Music, YouTube Music):
- Người dùng nghe nhạc theo bài hát, tạo playlist theo từng bài hát cụ thể, chứ không nghe toàn bộ mọi tác phẩm của một nghệ sĩ.
- Một nghệ sĩ có thể có nhiều phong cách âm nhạc hoặc bài hát khác nhau; việc chỉ gợi ý tên nghệ sĩ không đáp ứng được tính cá nhân hóa sâu sắc của một hệ thống Trí tuệ Nhân tạo hiện đại.

Do đó, **Last.fm 2K HetRec 2011 chính thức không còn là tập dữ liệu chính**. Thay vào đó, dự án chuyển đổi sang tập dữ liệu chuẩn mực cấp bài hát: **Million Song Dataset (MSD) – Taste Profile Subset**.

---

## 2. BẢNG SO SÁNH THỰC THỂ GỢI Ý: ARTIST-LEVEL VS SONG-LEVEL

| Tiêu chí | Cấu hình cũ Phase M0 (Last.fm 2K) | Cấu hình chuẩn hóa mới Phase M1 (MSD Taste Profile) |
|---|---|---|
| **Thực thể gợi ý chính (Primary Entity)** | Nghệ sĩ (`artist_id`) | **Bài hát (`song_id`)** |
| **Cặp tương tác nguyên tử** | `(user_id, artist_id, weight)` | **`(user_id, song_id, play_count)`** |
| **Vai trò của Nghệ sĩ (Artist Role)** | Mục tiêu dự đoán và xếp hạng | **Metadata bổ trợ đi kèm với bài hát (`artist_name`)** |
| **Vai trò của Thể loại & Album** | Thẻ tag gán cho nghệ sĩ | **Metadata thuộc tính của bài hát (`title`, `track_id`)** |
| **Trải nghiệm giao diện (UI)** | Thẻ nghệ sĩ (Artist Card) | **Thẻ bài hát (Song Card) kèm nút Play, thời lượng, sóng âm** |
| **Bài toán xếp hạng Top-K** | Xếp hạng Top-$K$ nghệ sĩ | **Xếp hạng Top-$K$ bài hát người dùng muốn nghe tiếp theo** |
| **Độ thưa thớt ma trận (Sparsity)** | 99.72% ($1.8K \times 17.6K$) | **> 99.8% (Đòi hỏi thuật toán tối ưu hóa cao hơn)** |

---

## 3. TẠI SAO CHỌN MILLION SONG DATASET (TASTE PROFILE SUBSET)?

Tập dữ liệu **Taste Profile Subset** (do The Echo Nest & Đại học Columbia LabROSA phối hợp công bố) là tiêu chuẩn vàng (de facto benchmark) của giới nghiên cứu khoa học RecSys quốc tế cho bài toán gợi ý phản hồi ngầm định mức bài hát:

1. **Tương tác thực 100%:** Chứa 48,373,586 bộ ba tương tác thật `(user_id, song_id, play_count)` từ hơn 1 triệu người dùng thực tế.
2. **Metadata phong phú và nhất quán:** Tệp ánh xạ chuẩn `unique_tracks.txt` liên kết chặt chẽ từng `song_id` với `track_id`, tên nghệ sĩ (`artist_name`) và tiêu đề bài hát (`song_title`).
3. **Tính tái lập khoa học (Reproducibility):** Được sử dụng chính thức trong cuộc thi *Kaggle Million Song Dataset Challenge* và hàng ngàn bài báo tại các hội nghị ACM RecSys, SIGIR, KDD.
4. **Giấy phép học thuật:** Công bố miễn phí cho mục đích nghiên cứu và giáo dục đại học.

---

## 4. CHIẾN LƯỢC TRÍCH XUẤT TẬP CON TÁI LẬP (REPRODUCIBLE SUBSET) TRÊN MÁY CÁ NHÂN

Do toàn bộ tập dữ liệu Taste Profile chứa 48 triệu tương tác (~2.9 GB tệp thô), việc huấn luyện trên laptop cá nhân (RAM 8GB–16GB) cần một chiến lược trích xuất mẫu khoa học, kiểm soát được kích thước bộ nhớ:

```text
[ Toàn bộ Taste Profile: 48M Triplets (2.9 GB) ]
                     │
                     ▼ (Streaming Chunk Filter: seed=42)
[ Core Subset Lọc Phân Tầng ]
  - Số lượng người dùng mục tiêu: 2,000 – 5,000 người dùng
  - Số lượng bài hát mục tiêu: 5,000 – 10,000 bài hát
  - Ngưỡng tương tác tối thiểu: >= 15 bài/người dùng; >= 10 người nghe/bài hát
                     │
                     ▼
[ Tệp Tương Tác Sạch: interactions.csv (~5 - 10 MB) ]
  - Hoàn toàn vừa vặn với RAM (< 200 MB khi huấn luyện)
  - Giữ nguyên 100% tính chất tương tác thật, không giả lập
```

---

## 5. NGUYÊN TẮC GIỮ NGUYÊN DẤU VẾT KIỂM TOÁN (AUDIT TRAIL)

Theo chuẩn mực kỹ thuật phần mềm:
- Không chỉnh sửa hồi tố (silent rewrite) các kết luận lịch sử trong các tệp tài liệu `docs/musicai/phase_m0_*.md`.
- Tài liệu này (`docs/musicai/phase_m1_song_level_decision.md`) đóng vai trò là **Bản tu chính án chính thức (Governing Amendment)**, có hiệu lực pháp lý cao nhất cho toàn bộ các giai đoạn từ Phase M1 đến Phase M6.
- Mọi mô hình, đường ống dữ liệu, API và giao diện của MusicAI từ thời điểm này sẽ chỉ phục vụ bài toán **Gợi ý Bài hát (Song Recommendation)**.
