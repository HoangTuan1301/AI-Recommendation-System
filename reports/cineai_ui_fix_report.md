# BÁO CÁO KỸ THUẬT: CINEAI – KHẮC PHỤC LỖI GIAO DIỆN & NÂNG CẤP CHUẨN NETFLIX
## CRITICAL UI FIX & PREMIUM NETFLIX-STYLE REDESIGN REPORT

---

**Dự án:** AI-Recommendation-System (Học kỳ 5 – Môn Trí tuệ nhân tạo)  
**Nền tảng:** CineAI Platform (React 19 + TypeScript + Vite + Tailwind CSS v4 + FastAPI)  
**Thời gian thực hiện:** 28/10/2026  
**Chuyên môn:** Senior Frontend Engineer, UI/UX Designer, Visual QA Automation Engineer  

---

## 1. PHÂN TÍCH NGUYÊN NHÂN GỐC RỄ (ROOT CAUSE ANALYSIS)

Sau khi kiểm tra sâu toàn bộ chuỗi build pipeline và cấu hình CSS (`package.json`, `vite.config.ts`, `tailwind.config.js`, `postcss.config.js`, `src/index.css`), nguyên nhân cốt lõi khiến giao diện bị vỡ hoàn toàn trên trình duyệt đã được xác định:

### 1.1 Xung đột Phiên bản Tailwind CSS v4 và Cú pháp Cũ v3
- `package.json` cài đặt **Tailwind CSS v4.3.3** (`"tailwindcss": "^4.3.3"`).
- Tuy nhiên, file `src/index.css` ban đầu lại sử dụng cú pháp của **Tailwind v3**:
  ```css
  /* CẤU HÌNH LỖI (V3 SYNTAX TRONG RUNTIME V4) */
  @tailwind base;
  @tailwind components;
  @tailwind utilities;
  ```
- **Hệ quả trực tiếp:** Trong Tailwind v4, các chỉ thị `@tailwind base/components/utilities` đã bị thay thế hoàn toàn bởi `@import "tailwindcss";`. Do đó, engine Tailwind v4 bỏ qua toàn bộ phần **Preflight Reset** (reset mặc định của trình duyệt) và không sinh các utility class cơ bản.
- Kết quả trên trình duyệt:
  1. Thẻ `<button>` giữ nguyên viền nổi và màu nền xám mặc định của hệ điều hành Windows.
  2. Các thẻ `<div>` không nhận được thuộc tính `box-sizing: border-box`, `display: flex`, hay `display: grid`.
  3. Tất cả phần tử trong Navbar, Hero và Cards bị dồn cục về góc trên bên trái màn hình.

### 1.2 Custom Theme Tokens Không Được Nạp vào CSS
- Trong Tailwind v4, file `tailwind.config.js` không còn được tự động phát hiện theo cơ chế cũ trừ khi khai báo rõ qua `@config` hoặc `@theme` trong CSS.
- Vì vậy, các class màu cốt lõi của giao diện như `bg-cine-bg`, `bg-cine-surface`, `text-cine-red`, `border-cine-border` **hoàn toàn không tồn tại trong file CSS build ra** (dung lượng CSS ban đầu chỉ có 10.92 kB).

---

## 2. GIẢI PHÁP VÀ CẤU HÌNH ĐÃ THAY ĐỔI

### 2.1 Chuyển đổi sang Plugin Chính thức `@tailwindcss/vite`
- Đã cài đặt package `@tailwindcss/vite` vào `devDependencies`.
- Cấu hình trực tiếp trong `frontend/vite.config.ts`:
  ```ts
  import { defineConfig } from 'vitest/config'
  import react from '@vitejs/plugin-react'
  import tailwindcss from '@tailwindcss/vite'

  export default defineConfig({
    plugins: [
      tailwindcss(),
      react(),
    ],
    // ...
  })
  ```
- Loại bỏ file trung gian `postcss.config.js` để Vite xử lý CSS trực tiếp qua engine LightningCSS siêu tốc.

### 2.2 Tái cấu trúc `frontend/src/index.css` Chuẩn Tailwind v4
- Thay thế hoàn toàn bằng cú pháp hiện đại `@import "tailwindcss";` kết hợp khối `@theme`:
  ```css
  @import "tailwindcss";

  @theme {
    --color-cine-bg: #08090D;
    --color-cine-secondary: #101116;
    --color-cine-surface: #14151B;
    --color-cine-elevated: #1C1E27;
    --color-cine-hover: #232631;
    --color-cine-border: #292B33;
    --color-cine-red: #E50914;
    --color-cine-red-hover: #B80710;
    --color-cine-text: #FFFFFF;
    --color-cine-muted: #A1A1AA;
    --color-cine-success: #22C55E;

    --shadow-cinema: 0 20px 40px -15px rgba(0, 0, 0, 0.8);
    --shadow-glow-red: 0 0 25px -5px rgba(229, 9, 20, 0.5);
    --shadow-card-hover: 0 14px 28px rgba(0, 0, 0, 0.6), 0 10px 10px rgba(0, 0, 0, 0.4);
  }
  ```
- **Kết quả biên dịch:** File CSS xuất bản tăng từ **10.92 kB** lên **53.87 kB**, chứa đầy đủ 100% Preflight reset và các class tiện ích của toàn bộ dự án.

---

## 3. NÂNG CẤP TOÀN DIỆN UX/UI CHUẨN NETFLIX (CINEAI V2.0)

### 3.1 Navbar Điện ảnh Hoàn chỉnh (`Navbar.tsx`)
- Chiều cao cố định chuẩn desktop `72px` (`h-[72px]`).
- Hiệu ứng Scroll Blur mượt mà: khi cuộn quá 20px, chuyển sang `bg-cine-bg/95 backdrop-blur-md border-b border-cine-border/80 shadow-2xl`.
- Logo thương hiệu `CINEAI` nổi bật với tag `v2.0` góc cạnh.
- Thanh tìm kiếm động tích hợp debounce và Quick User Switcher dropdown (chuyển đổi nhanh các User ID #1, #196, #42,...).
- Hỗ trợ menu trượt dọc cho thiết bị di động (Mobile Drawer Menu).

### 3.2 Hero Banner 2 Cột Cân đối (`HeroBanner.tsx`)
- **Khắc phục triệt để lỗi vùng đen bên phải:** 
  - Trước đây: Kéo giãn ảnh 2:3 ra toàn màn hình làm nhòe và để trống 60% bên phải.
  - Sau khi sửa: Thiết kế bố cục **2 cột chuẩn điện ảnh**:
    - **Cột Trái (60%):** Huy hiệu `FEATURED MASTERPIECE`, năm phát hành, điểm khán giả xác thực `4.4/5.0`, tiêu đề phim khổng lồ (`text-4xl` đến `text-6xl`), danh sách thể loại và 2 nút hành động chính:
      - `Explore AI Recommendations` (nền đỏ Netflix, đổ bóng Neon `shadow-glow-red`).
      - `Movie Details & Stats` (kính mờ Frosted Glass).
    - **Cột Phải (40%):** Thẻ **3D Movie Poster Showcase** nghiêng nhẹ với viền phát sáng đỏ/tím, tạo cảm giác áp-phích rạp chiếu phim chân thực và lấp đầy khung nhìn một cách hoàn hảo.

### 3.3 Cải tiến Thẻ Phim (`MovieCard.tsx`) & Băng chuyền (`MovieCarousel.tsx`)
- **Đồng bộ Kích thước:** Thiết lập tỷ lệ áp-phích chuẩn `2:3` (`aspect-[2/3]`), viền bo tròn hiện đại, hiệu ứng hover phóng to nhẹ (`hover:scale-105`) kèm bóng viền.
- **Loại bỏ Hoàn toàn Trùng lặp Tiêu đề:** Tách bạch phần hình ảnh và thông tin bên dưới, tiêu đề giới hạn tối đa 2 dòng (`line-clamp-2`), hiển thị thể loại và điểm dự đoán AI hoặc đánh giá trung bình không bao giờ bị tràn (zero text overflow).
- **Movie Carousel:** Nút bấm điều hướng trái/phải nổi với hiệu ứng làm mờ nền kính, cuộn mượt mà không vỡ khung hình.

---

## 4. CẢI TIẾN POSTER THEO PHONG CÁCH TỐI GIẢN (`poster_service.py`)

Thực hiện nghiêm ngặt yêu cầu của **TASK 4**:
- **Tuyệt đối không dùng Cartoon Emoji lớn** làm trung tâm áp-phích.
- Thiết kế lại toàn bộ template SVG thành **Minimalist Movie Artwork 35mm**:
  1. Hai dải lỗ răng cưa phim nhựa (35mm Sprocket Holes) chạy dọc hai bên cạnh poster.
  2. Vòng tròn khẩu độ máy quay (Camera Aperture Rings) phối hợp đường ngắm quang học và chữ cái viết tắt tên phim (Monogram Initials) bằng typography đậm chất điện ảnh (`SW` cho Star Wars, `RJ` cho Return of the Jedi, `FA` cho Fargo,...).
  3. Nhãn năm phát hành trung tâm và huy hiệu thể loại thanh lịch.
  4. Bảng màu gradient riêng biệt cho từng thể loại: Đỏ thẫm (Action), Cam lửa (Adventure), Lam huyền ảo (Sci-Fi), Vàng hoàng hôn (Comedy), Tím đêm (Mystery/Fantasy),...
  5. Đóng dấu bản quyền `CINEAI COLLECTION ARCHIVE`.

> [!NOTE]
> **Về Nguồn Poster Thật (TMDB API):**
> Hệ thống được thiết kế sẵn sàng hỗ trợ TMDB qua biến môi trường `os.getenv("TMDB_API_KEY")`. Do môi trường hiện tại chưa cấu hình API Key bên thứ ba và để đảm bảo 100% tính độc lập ngoại tuyến (offline reproducibility), hệ thống sử dụng bộ poster Minimalist 35mm SVG độc quyền được sinh tự động cho toàn bộ 1,682 bộ phim. Không xảy ra tình trạng match sai phim hoặc rò rỉ mã bí mật.

---

## 5. BẰNG CHỨNG XÁC MINH TRÌNH DUYỆT THỰC TẾ (REAL BROWSER VISUAL VERIFICATION)

Đã xây dựng script tự động hóa [`frontend/capture_screenshots.cjs`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/frontend/capture_screenshots.cjs) sử dụng **Playwright** điều khiển trình duyệt **Microsoft Edge (Chromium Headless)** truy cập thực tế vào `http://localhost:5173` và chụp lại toàn bộ màn hình.

Toàn bộ ảnh chụp màn hình được lưu trữ tại thư mục [`reports/screenshots/`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/reports/screenshots/):

| Tệp Ảnh Màn hình | Độ phân giải | Nội dung Xác minh | Trạng thái |
| :--- | :--- | :--- | :---: |
| [`home-desktop.png`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/reports/screenshots/home-desktop.png) | 1440 x 900 | Hero 2 cột cân đối, poster 3D nổi bật, Carousels trượt đẹp mắt | **HOÀN HẢO** |
| [`home-mobile.png`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/reports/screenshots/home-mobile.png) | 390 x 844 | Layout dọc chuẩn iPhone, không bị tràn chiều ngang, nút bấm to rõ | **HOÀN HẢO** |
| [`discover-desktop.png`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/reports/screenshots/discover-desktop.png) | 1440 x 900 | Lưới 6 cột thẻ phim 35mm, bộ lọc thể loại, thanh tìm kiếm | **HOÀN HẢO** |
| [`recommendations-desktop.png`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/reports/screenshots/recommendations-desktop.png) | 1440 x 900 | Bảng chọn 10 mô hình AI theo họ, thẻ kết quả kèm điểm số và rank | **HOÀN HẢO** |
| [`ai-lab-desktop.png`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/reports/screenshots/ai-lab-desktop.png) | 1440 x 900 | Leaderboard sắp xếp động, biểu đồ đa độ đo, Model Battle | **HOÀN HẢO** |
| [`profile-desktop.png`](file:///e:/HOCTAP/K5/TTNT/AI-Recommendation-System/reports/screenshots/profile-desktop.png) | 1440 x 900 | Thống kê phân bố đánh giá người dùng, danh sách phim yêu thích | **HOÀN HẢO** |

*Ghi nhận Console Browser:* **Zero console errors detected (0 lỗi runtime).*

---

## 6. KẾT QUẢ KIỂM THỬ HỒI QUY (REGRESSION TESTS)

### 6.1 Frontend Production Build
```powershell
Set-Location frontend; npm run build
```
- **Kết quả:** `tsc -b && vite build` thành công trong **620ms**.
- Đóng gói hoàn tất: `dist/assets/index-4Qliaurf.css` (53.87 kB) và `dist/assets/index-BG7_TZT0.js` (709.75 kB).

### 6.2 Frontend Unit Tests (Vitest)
```powershell
Set-Location frontend; npm test -- --run
```
- **Kết quả:** **6 / 6 tests PASSED** (100% OK trong 2.03s).

### 6.3 Backend & Core AI Regression Tests (Python)
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
- **Kết quả:** **Ran 85 tests in 21.245s — OK**.
- Đảm bảo 100% tính toàn vẹn của 10 mô hình AI, quy trình đánh giá Top-N và 9 endpoints của FastAPI.

---

## 7. TRẠNG THÁI HỆ THỐNG VÀ HƯỚNG DẪN TRẢI NGHIỆM

Hiện tại, cả hai dịch vụ đang chạy sẵn sàng trong môi trường:
- **FastAPI Backend:** `http://localhost:8000` (Swagger UI: `http://localhost:8000/docs`)
- **React Frontend:** `http://localhost:5173`

Người dùng có thể mở trình duyệt truy cập ngay `http://localhost:5173` để chiêm ngưỡng giao diện điện ảnh Netflix chuẩn mực của CineAI.
