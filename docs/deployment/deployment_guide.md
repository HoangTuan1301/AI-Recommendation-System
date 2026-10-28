# HƯỚNG DẪN TRIỂN KHAI HỆ THỐNG ĐA MIỀN (DEPLOYMENT GUIDE)

**Hệ thống:** CineAI V2.0 + MusicAI  
**Phiên bản:** 1.0  
**Giai đoạn:** Phase M6  
**Trạng thái Triển khai Staging / Production:** **NOT EXECUTED** (Chuẩn bị sẵn sàng hạ tầng, chưa deploy lên dịch vụ bên ngoài nhằm tuân thủ nguyên tắc không tự ý phát sinh chi phí).

---

## 1. PHÂN TÍCH TÀI NGUYÊN VÀ KHẢ THI TRIỂN KHAI (RESOURCE SIZING)

Dựa trên kết quả đo đạc thực tế từ quá trình benchmark hiệu năng và kiểm toán artifacts:

| Thành phần | Mức tiêu thụ thực tế | Khuyến nghị tối thiểu (Minimum) | Khuyến nghị sản xuất (Production) |
|:---|:---:|:---:|:---:|
| **RAM Backend** | ~61.5 MB RSS | 512 MB RAM | 1 GB — 2 GB RAM |
| **RAM Frontend (Nginx)** | ~15 MB RSS | 128 MB RAM | 256 MB RAM |
| **CPU Usage** | < 5% khi nhàn rỗi, ~20% khi suy luận | 1 vCPU | 2 vCPU |
| **Dung lượng Checkpoints (16 mô hình)** | 70.3 MB | 200 MB ổ đĩa | 500 MB ổ đĩa |
| **Dung lượng Dữ liệu đã xử lý** | ~5.8 MB | 50 MB ổ đĩa | 100 MB ổ đĩa |
| **Thời gian khởi động (Cold Startup)** | ~1.8 giây | < 10 giây | < 5 giây |

> [!NOTE]
> Nhờ chiến lược **Lazy Loading** kết hợp với **Single-Thread Matrix Factorization**, toàn bộ hệ thống gợi ý 16 mô hình chỉ tiêu thụ chưa đầy **100 MB RAM**, hoàn toàn có thể triển khai mượt mà trên các gói máy chủ đám mây giá rẻ (e.g. AWS t4g.micro/small, GCP e2-micro/small, Render Free/Starter, DigitalOcean 1GB Droplet).

---

## 2. CÁC PHƯƠNG ÁN TRIỂN KHAI ĐƯỢC ĐỀ XUẤT

### Phương án A: Triển khai Độc lập bằng Docker Compose trên Virtual Private Server (VPS / Cloud VM)
- **Nhà cung cấp phù hợp:** DigitalOcean, Hetzner, AWS EC2, GCP Compute Engine, Azure VM.
- **Quy trình thực hiện:**
  1. Clone repository về VPS:
     ```bash
     git clone <repo_url>
     cd AI-Recommendation-System
     ```
  2. Sao chép các tệp checkpoint và processed data vào đúng thư mục:
     - `models/checkpoints/` (10 checkpoint CineAI)
     - `models/checkpoints/music/` (6 checkpoint MusicAI)
     - `data/processed/`
     - `data/music/processed/`
  3. Xác thực artifacts:
     ```bash
     python scripts/verify_production_artifacts.py
     ```
  4. Cấu hình HTTPS / Tên miền (Domain SSL):
     - Sử dụng Nginx Reverse Proxy bên ngoài hoặc Traefik / Caddy tự động cấp chứng chỉ Let's Encrypt SSL.
  5. Khởi động hệ thống:
     ```bash
     docker compose up -d --build
     ```

### Phương án B: Triển khai theo kiến trúc Container PaaS (Render / Railway / Fly.io / GCP Cloud Run)
- **Backend:** Đóng gói image Docker từ `Dockerfile.backend`, kết nối Persistent Disk để gắn `models/checkpoints` và `data/processed`.
- **Frontend:** Triển khai dưới dạng Static Site trên Vercel, Netlify hoặc Cloudflare Pages; trỏ API rewrite rule `/api/*` về địa chỉ domain của Backend.

---

## 3. CHECKLIST KIỂM ĐỊNH SẴN SÀNG TRIỂN KHAI (GO-LIVE CHECKLIST)

- [x] Đã tối ưu hóa code-splitting bundle frontend (Main bundle < 350 KB).
- [x] Đã cấu hình non-root user cho cả Backend container (`appuser`) và Frontend container (`nginx`).
- [x] Đã bổ sung 4 Security Headers bắt buộc trên Nginx.
- [x] Đã kiểm tra 0 secrets bị commit vào Git.
- [x] Đã xác minh tính toàn vẹn 16/16 checkpoints (SHA-256 match).
- [x] Đã xác minh 22/22 kịch bản E2E kiểm thử thành công trên trình duyệt thật.
- [x] Đã xác minh 134/134 bài kiểm thử Pytest đạt 100%.
- [x] Đã ghi nhận thời gian đáp ứng API P95 < 15ms.
- [ ] Phê duyệt triển khai từ người dùng (Chờ phê duyệt trước khi kích hoạt Cloud Staging).
