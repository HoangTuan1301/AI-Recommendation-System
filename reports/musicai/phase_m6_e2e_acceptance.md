# BÁO CÁO KIỂM THỬ THỰC TẾ PLAYWRIGHT E2E — PHASE M6

**Hệ thống:** CineAI V2.0 + MusicAI  
**Môi trường kiểm thử:** Microsoft Edge (Chromium Engine), Headless Mode, Viewports: Desktop (1440x900), Tablet (768x1024), Mobile (375x812)  
**Máy chủ phục vụ:** FastAPI Uvicorn Server (Port 8000), Vite Preview Server (Port 5173)  
**Ngày kiểm thử:** 28/10/2026  
**Tệp kết quả JSON:** `reports/musicai/phase_m6_e2e_results.json`  

---

## 1. TỔNG KẾT KẾT QUẢ CHẠY E2E

```
================================================================
  PHASE M6 E2E AUDIT SUMMARY
  Total Scenarios Executed: 22
  Passed Scenarios:         22 (100%)
  Failed Scenarios:          0 (0%)
  Console Errors Logged:     0
  Network Errors Logged:     0
================================================================
```

---

## 2. KẾT QUẢ CHI TIẾT THEO 15 KỊCH BẢN BẮT BUỘC

| Kịch bản | Hạng mục kiểm thử | Kết quả thực tế | Trạng thái |
|:---:|:---|:---|:---:|
| **1** | Mở CineAI và xem danh sách phim | Logo `CINEAI` hiển thị, trang `/discover` tải danh sách phim từ API thật. | **PASS** |
| **2** | Chọn mô hình CineAI và nhận gợi ý phim | Trang `/recommendations` suy luận thời gian thực, trả về 10 phim kèm badge `#1`. | **PASS** |
| **3** | Chuyển từ CineAI sang MusicAI | Click "Music" trên Domain Switcher, logo chuyển thành `MUSICAI`, theme đổi sang ngọc lục bảo. | **PASS** |
| **4** | Chọn người dùng MusicAI và nhận Top-10 | Chọn Demo User A (`0023d50b...`), trả về 10 bài hát gợi ý có đầy đủ rank từ `#1` đến `#10`. | **PASS** |
| **5** | Chuyển đổi giữa cả 6 mô hình MusicAI | Lần lượt suy luận: `User-Based CF`, `Implicit ALS`, `BPR-MF`, `Neural CF`, `Popularity`, `Item-Based CF`. Cả 6 mô hình đều trả về danh sách bài hát mới thành công. | **PASS** |
| **6** | Kiểm tra tên bài hát và nghệ sĩ từ API thật | Xác nhận bài hát có tên, nghệ sĩ và điểm tương thích (`score`) chuẩn định dạng từ Million Song Dataset. | **PASS** |
| **7** | Xem thông tin chi tiết bài hát | Mở Song Detail Modal: hiển thị `Song ID (MSD)`, `Track ID (Echo Nest)`, `Total Listens (Train)`; đóng modal sạch sẽ. | **PASS** |
| **8** | Xem lịch sử nghe của người dùng demo | Mở User Listening History Modal: hiển thị 18 bài hát người dùng đã nghe trong tập huấn luyện `train.csv`. | **PASS** |
| **9** | Kiểm tra loại trừ trùng lặp (Deduplication) | Đối chiếu lịch sử nghe và danh sách đề xuất: **0 bài hát trùng lặp (Overlap count = 0)**. | **PASS** |
| **10** | Kiểm tra Cold-Start Fallback | Nhập user mới chưa từng xuất hiện: banner màu vàng hổ phách hiển thị "Cold-Start Fallback Active", tự động chuyển sang mô hình `popularity` và trả về 10 bài hát. | **PASS** |
| **11** | Kiểm tra phục hồi từ Fallback | Chọn lại User A có trong tập huấn luyện: hệ thống tự động thoát khỏi chế độ Fallback và quay lại gợi ý cá nhân hóa. | **PASS** |
| **12** | Chuyển từ MusicAI về lại CineAI | Click "Movies" trên Domain Switcher: giao diện trở lại CineAI với theme đỏ Netflix và logo `CINEAI`. | **PASS** |
| **13** | Kiểm tra Responsive Layout | - Tablet (768x1024): Lưới bài hát thích ứng mượt mà (10 cards).<br>- Mobile (375x812): Hiển thị dạng dọc tối ưu cảm ứng (10 cards). | **PASS** |
| **14** | Kiểm tra Điều hướng URL Trực tiếp | Truy cập trực tiếp qua deep-links: `/music`, `/music/models`, `/lab`. Toàn bộ dữ liệu hiển thị chính xác mà không bị mất ngữ cảnh. | **PASS** |
| **15** | Kiểm tra Reload trang và Lưu trạng thái | F5 reload tại `/music/recommendations`: trạng thái miền Music và danh mục bài hát được duy trì ổn định. | **PASS** |

---

## 3. DANH SÁCH ẢNH CHỤP MÀN HÌNH MINH CHỨNG (SCREENSHOTS)

Toàn bộ ảnh chụp màn hình được lưu trữ tại thư mục: [`reports/screenshots/e2e_m6/`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/reports/screenshots/e2e_m6):
- `01_cineai_catalog.png`: Danh mục phim CineAI Discover.
- `02_cineai_recommendations.png`: Gợi ý phim với huy hiệu thứ hạng #1.
- `03_domain_switch_to_music.png`: Giao diện MusicAI sau khi chuyển đổi miền.
- `04_musicai_top10_songs.png`: Top-10 bài hát gợi ý từ Item-Based CF.
- `05_all_6_models.png`: Bộ chọn và kết quả của cả 6 mô hình AI âm nhạc.
- `07_song_detail_modal.png`: Modal chi tiết bài hát với mã MSD và Echo Nest.
- `08_user_history_modal.png`: Modal lịch sử nghe nhạc trong tập huấn luyện.
- `10_cold_start_fallback.png`: Banner cảnh báo Cold-Start Fallback tự động.
- `12_switch_back_to_cineai.png`: Giao diện quay trở lại CineAI.
- `13_tablet_viewport.png`: Hiển thị trên thiết bị máy tính bảng.
- `13_mobile_viewport.png`: Hiển thị trên thiết bị di động.
- `15_page_reload_persistence.png`: Kiểm chứng tính bền vững của dữ liệu sau reload.
