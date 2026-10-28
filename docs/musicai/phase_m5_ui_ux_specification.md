# MUSICAI — PHASE M5: UI/UX DESIGN & INTERACTION SPECIFICATION

**Tài liệu:** Đặc tả Thiết kế Giao diện và Trải nghiệm Người dùng (UI/UX Specification)  
**Phiên bản:** 1.0  
**Giai đoạn:** Phase M5  
**Hệ thống:** CineAI V2.0 + MusicAI  

---

## 1. NGUYÊN TẮC THIẾT KẾ ĐỒNG NHẤT (DESIGN SYSTEM)

Cả hai hệ thống chia sẻ chung một nền tảng Dark Theme điện ảnh / âm nhạc cao cấp:
- **Background chính:** `zinc-950` (`#09090b`) kết hợp với hiệu ứng gradient đa tầng.
- **Surface / Card Background:** `zinc-900/90` (`#18181b`) với border mờ `border-zinc-800`.
- **Typography:** Inter sans-serif tiêu chuẩn quốc tế, hỗ trợ tiếng Anh và tiếng Việt mượt mà.

### Bảng màu nhận diện thương hiệu theo Miền (Domain Color Language):
| Thuộc tính | CineAI (Movies) | MusicAI (Songs) |
|:---|:---|:---|
| **Màu chủ đạo (Primary)** | Đỏ Netflix (`#E50914` / `rose-600`) | Xanh ngọc lục bảo (`#10B981` / `emerald-500`) |
| **Màu phụ trợ (Secondary)**| Vàng kim (`#EAB308` / `amber-500`) | Xanh mòng két (`#14B8A6` / `teal-500`) |
| **Hiệu ứng Glow (Glow Shadow)**| `rgba(229, 9, 20, 0.35)` | `rgba(16, 185, 129, 0.35)` |
| **Biểu tượng nhận diện** | `Film`, `Clapperboard`, `Popcorn` | `Music`, `Disc3`, `Headphones`, `Sparkles` |

---

## 2. ĐẶC TẢ CHI TIẾT CÁC THÀNH PHẦN GIAO DIỆN

### 2.1. Thanh điều hướng Đa miền (Multi-Domain Navbar)
- **Logo động:**
  - Khi ở chế độ Movies: Hiển thị `CINE` + badge đỏ `AI`.
  - Khi ở chế độ Music: Hiển thị biểu tượng `Music` ngọc lục bảo + `MUSIC` + badge xanh `AI`.
- **Domain Switcher:** Đặt ở vị trí trung tâm hoặc góc phải nổi bật. Thiết kế dạng trượt bóng (pill switch) chuyển đổi tức thời giữa Movies và Music.
- **Menu điều hướng theo miền:**
  - Menu Movies: `Discover`, `AI Recommendations`, `AI Lab & Benchmark`, `Model Comparison`.
  - Menu Music: `Music Home`, `Song Recommendations`, `Model Leaderboard`.
- **Quick Switcher người dùng:**
  - Cho phép chọn nhanh các User ID mẫu trong catalog để kiểm thử gợi ý tức thì mà không cần nhập tay.

### 2.2. Bảng điều khiển Gợi ý Bài hát (Music Recommendations Page)
- **User Selector Bar:**
  - 5 nút chip (pill buttons) đại diện cho các người dùng mẫu có gu âm nhạc đa dạng.
  - Ô nhập User ID tự do cho phép kiểm thử bất kỳ người dùng nào trong 2.520 người dùng huấn luyện hoặc người dùng mới hoàn toàn (Cold-Start test).
  - Nút xem "Listening History" mở drawer/modal danh sách bài hát người dùng đã nghe.
- **Model Selector Grid:**
  - 6 thẻ mô hình trực quan: Item-Based CF, User-Based CF, Implicit ALS, BPR-MF, Neural CF, Popularity.
  - Mỗi thẻ hiển thị: Phân loại thuật toán, độ phức tạp tính toán, thời gian suy luận (latency), mô tả khoa học ngắn gọn, và huy hiệu "DEFAULT" đối với Item-CF.
- **Top-K Selector:**
  - Nút chọn nhanh: 5, 10 (mặc định), 20 bài hát.
- **Banner Kết quả & Cảnh báo:**
  - Hiển thị tóm tắt: Model đã dùng, Thời gian phản hồi Backend (`latency_ms`), Số lượng bài hát được đề xuất.
  - **Cold-Start Fallback Alert:** Nếu người dùng chưa từng nghe nhạc (người dùng mới), hệ thống tự động fallback về mô hình `popularity` và hiển thị banner cảnh báo màu vàng hổ phách giải thích nguyên nhân rõ ràng và minh bạch cho người dùng.
- **Lưới thẻ bài hát (Song Grid):**
  - Hiển thị theo lưới 1 đến 5 cột (responsive từ mobile đến desktop 4K).
  - Mỗi bài hát hiển thị: Thứ hạng, Điểm tương thích (`score`), Tên bài hát, Nghệ sĩ, Mã `song_id`, Biểu tượng vinyl quay khi hover, và nút mở Modal chi tiết.

### 2.3. Modal Chi tiết Bài hát (Song Detail Modal)
- Nổi bật giữa màn hình với phông nền mờ ảo (Backdrop blur).
- Cung cấp dữ liệu chi tiết từ Echo Nest và Million Song Dataset:
  - Tên tác phẩm và Nghệ sĩ biểu diễn.
  - Mã nhận diện song_id (MSD) và track_id (Echo Nest).
  - Tổng số lượt nghe và số lượng thính giả trong tập huấn luyện.
  - Trạng thái kiểm chứng khoa học.

### 2.4. Drawer Lịch sử Nghe nhạc (Listening History Modal)
- Danh sách dạng bảng/thẻ hiển thị toàn bộ các bài hát người dùng đã nghe trong tập huấn luyện `train.csv`.
- Mỗi dòng hiển thị thứ tự, tên bài hát, nghệ sĩ, và mã định danh.
- Đóng vai trò là công cụ kiểm toán trực quan (Visual Audit Tool) để người dùng xác minh:
  1. Các bài hát gợi ý có phản ánh đúng sở thích trong lịch sử hay không.
  2. Không có bài hát nào đã nghe bị gợi ý lại (Zero Duplication Rule).

---

## 3. KHẢ NĂNG TIẾP CẬN VÀ TRẢI NGHIỆM ĐỒNG BỘ (ACCESSIBILITY & RESPONSIVENESS)

- **Responsive Breakpoints:** Tối ưu mượt mà từ mobile 375px, tablet 768px, laptop 1024px đến màn hình máy tính để bàn 1440px+.
- **Accessibility:** Các nút bấm, switcher, modal đều có thuộc tính ARIA, hỗ trợ bàn phím (Escape để đóng modal), độ tương phản văn bản đạt chuẩn WCAG AA+.
- **Feedback & Loading States:** Biểu tượng xoay loading thanh thoát khi gửi request tới API, thông báo lỗi thân thiện khi backend không phản hồi hoặc mất kết nối mạng.
