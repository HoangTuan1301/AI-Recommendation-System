# HƯỚNG DẪN KẾT NỐI VÀ TRIỂN KHAI FRONTEND LÊN VERCEL (VERCEL INTEGRATION GUIDE)

**Dự án:** Multi-Domain AI Recommendation System (CineAI V2.0 + MusicAI)  
**Công nghệ:** React 19 + TypeScript + Vite + Tailwind CSS  
**Mục tiêu:** Hướng dẫn cấu hình kết nối ứng dụng React trên Vercel tới Backend FastAPI trên Render.

---

## 1. CƠ CHẾ KẾT NỐI API GIỮA VERCEL VÀ RENDER

Ứng dụng Frontend React đã được cấu hình linh hoạt:
- Trong mã nguồn client (`frontend/src/services/api.ts` và `frontend/src/services/musicApi.ts`), hệ thống đọc biến môi trường:
  $$\text{API\_BASE} = \text{import.meta.env.VITE\_API\_BASE\_URL} + \text{"/api/v1"}$$
  $$\text{MUSIC\_API\_BASE} = \text{import.meta.env.VITE\_API\_BASE\_URL} + \text{"/api/music"}$$
- **Nếu `VITE_API_BASE_URL` được đặt:** Frontend sẽ gọi trực tiếp sang domain Render qua giao thức HTTPS có bảo mật CORS.
- **Nếu `VITE_API_BASE_URL` để trống:** Frontend mặc định sử dụng đường dẫn tương đối `/api/v1` và `/api/music` (phù hợp với môi trường local dev proxy hoặc Docker Nginx).

---

## 2. XỬ LÝ LỖI 404 TRÊN SINGLE PAGE APPLICATION (SPA) KHI RELOAD TRANG

Khi người dùng truy cập trực tiếp các route con (ví dụ: `/music/recommendations`, `/discover`, `/lab`) hoặc nhấn phím F5 / Reload, server Vercel nếu không có file `vercel.json` sẽ tìm tệp vật lý tại đường dẫn đó và trả về mã lỗi **404 Not Found**.

Để khắc phục triệt để, tệp [`frontend/vercel.json`](file:///E:/HOCTAP/K5/TTNT/AI-Recommendation-System-Clean/frontend/vercel.json) đã được tích hợp sẵn quy tắc viết lại đường dẫn (URL Rewrite):
```json
{
  "rewrites": [
    {
      "source": "/(.*)",
      "destination": "/index.html"
    }
  ],
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        { "key": "X-Content-Type-Options", "value": "nosniff" },
        { "key": "X-Frame-Options", "value": "SAMEORIGIN" },
        { "key": "X-XSS-Protection", "value": "1; mode=block" }
      ]
    }
  ]
}
```

---

## 3. CÁC BƯỚC CẤU HÌNH TRÊN VERCEL DASHBOARD

### Cách 1: Dự án đã được tạo sẵn trên Vercel (Cập nhật biến môi trường)
1. Đăng nhập vào [Vercel Dashboard](https://vercel.com/dashboard).
2. Chọn dự án Frontend đang chạy của bạn.
3. Chuyển sang tab **Settings** $\rightarrow$ Chọn mục **Environment Variables** ở menu bên trái.
4. Thêm biến môi trường mới:
   - **Key:** `VITE_API_BASE_URL`
   - **Value:** `https://cineai-musicai-backend.onrender.com` *(Thay bằng URL Render thật của bạn, không có dấu gạch chéo `/` ở cuối)*
   - **Environment:** Tích chọn cả 3 mục: **Production**, **Preview**, **Development**.
5. Bấm **Save**.
6. **Quan trọng:** Sau khi lưu biến môi trường, chuyển sang tab **Deployments**, bấm vào dấu ba chấm `...` của bản build gần nhất và chọn **Redeploy** (đảm bảo không tích chọn "Use existing Build Cache") để Vite nhúng biến môi trường mới vào bundle JS.

---

### Cách 2: Triển khai dự án mới từ đầu trên Vercel
1. Tại Vercel Dashboard, bấm **Add New...** $\rightarrow$ Chọn **Project**.
2. Chọn repository GitHub: `https://github.com/HoangTuan1301/AI-Recommendation-System`.
3. Cấu hình thông số Project:
   - **Framework Preset:** `Vite`
   - **Root Directory:** Bấm **Edit** và chọn thư mục `frontend`
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
   - **Install Command:** `npm install`
4. Mở rộng mục **Environment Variables**:
   - Thêm `VITE_API_BASE_URL` = `https://cineai-musicai-backend.onrender.com`
5. Bấm nút **Deploy**.
6. Vercel sẽ tự động build và cấp domain dạng `https://ai-recommendation-system.vercel.app`.

---

## 4. XỬ LÝ TRẢI NGHIỆM NGƯỜI DÙNG KHI GẶP COLD-START CỦA RENDER

Do Render Free Tier có thời gian ngủ đông sau 15 phút không hoạt động, lần truy cập đầu tiên của giảng viên hoặc thành viên nhóm có thể mất **50–60 giây** để đánh thức container.

Ứng dụng Frontend đã được trang bị thông báo thân thiện:
- Khi request gặp trạng thái chờ hoặc tạm thời mất kết nối trong lúc khởi động nguội, giao diện sẽ hiển thị thông điệp cảnh báo rõ ràng:
  > *"Không thể kết nối tới máy chủ Backend. Máy chủ có thể đang khởi động lại (Render Cold Start mất khoảng 50-60s), vui lòng đợi giây lát và thử lại!"*
- Khi máy chủ khởi động xong, các lần gọi tiếp theo sẽ phản hồi cực nhanh trong vòng **10–25ms**.

---

## 5. BẢNG KIỂM TRA CHỨC NĂNG SAU KHI KẾT NỐI (SMOKE TEST)

Truy cập website trên trình duyệt máy tính và điện thoại để kiểm chứng:
- [ ] Trang chủ CineAI hiển thị đầy đủ Hero Banner và danh sách phim phổ biến.
- [ ] Bấm nút **Domain Switcher** (trên thanh Navbar): Giao diện chuyển mượt mà sang theme xanh Emerald của MusicAI.
- [ ] Trang gợi ý bài hát (`/music/recommendations`): Hiển thị đầy đủ Top-10 bài hát gợi ý từ mô hình `item_cf`.
- [ ] Thử nghiệm đổi sang các mô hình khác (`user_cf`, `ials`, `bpr_mf`, `implicit_neural_cf`, `popularity`): Danh sách bài hát cập nhật theo đúng thuật toán.
- [ ] Mở modal chi tiết bài hát và modal lịch sử nghe của user: Dữ liệu hiển thị chính xác từ Million Song Dataset.
- [ ] Thử nghiệm nhập `user_id = 999999999`: Hiển thị Cold-start banner cảnh báo người dùng mới và tự động đề xuất Top Popularity.
- [ ] Nhấn F5 / Reload trang: Trình duyệt duy trì đúng trang hiện tại, không bị lỗi 404.
