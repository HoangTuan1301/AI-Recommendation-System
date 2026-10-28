# DANH MỤC KIỂM TRA SẴN SÀNG TRIỂN KHAI PRODUCTION (PRODUCTION CHECKLIST)

**Dự án:** Multi-Domain AI Recommendation System (CineAI V2.0 + MusicAI)  
**Tác giả:** DevOps Lead & QA Lead  
**Quyết định phê duyệt:** Sẵn sàng Go-Live khi hoàn tất checklist.

---

## 1. BẢNG KIỂM TRA TRƯỚC TRIỂN KHAI (PRE-DEPLOYMENT CHECKLIST)

| Hạng mục kiểm tra | Tiêu chuẩn kỹ thuật bắt buộc | Trạng thái thực tế | Bằng chứng kiểm tra |
| :--- | :--- | :---: | :--- |
| **Clean Workspace** | Thực hiện duy nhất trên `AI-Recommendation-System-Clean` | **PASS** | Đường dẫn xác nhận: `E:\HOCTAP\K5\TTNT\AI-Recommendation-System-Clean` |
| **Git Safety** | Không chạy `reset --hard`, không xóa nhánh | **PASS** | Nhánh `main` đồng bộ với `origin/main` |
| **Python Version** | Python 3.11.x tương thích thư viện khoa học | **PASS** | Python 3.11.9 xác thực |
| **Backend Dependencies** | Không có xung đột phiên bản (`pip check`) | **PASS** | `No broken requirements found.` (đã bổ sung `pyarrow>=15.0.0`) |
| **Frontend Dependencies** | Không có lỗ hổng bảo mật (`npm audit`) | **PASS** | `audited 181 packages, found 0 vulnerabilities` |
| **Frontend Production Build** | TypeScript strict và Vite build thành công | **PASS** | `npm run build` thành công trong 1.37s |
| **Docker Non-Root** | Container thực thi với user không đặc quyền | **PASS** | `appuser` (UID 10001) trên Backend, `nginx` (UID 101) trên Frontend |
| **Bảo mật Bí mật (.env)** | Không commit tệp bí mật, API key vào Git | **PASS** | `.env` nằm trong `.gitignore`, `.env.example` chuẩn hóa |
| **Khống chế Tài nguyên** | Giới hạn đơn luồng CPU chống quá tải RAM | **PASS** | `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1` |
| **Bộ nhớ Cache Mô hình** | Giới hạn số mô hình đồng thời trên RAM cho Free Tier | **PASS** | Cơ chế LRU Eviction tích hợp sẵn qua `MAX_CACHED_MODELS=4` |

---

## 2. MA TRẬN TOÀN VẸN CHECKPOINT VÀ DỮ LIỆU (16 MÔ HÌNH AI)

Tất cả các mô hình và tập dữ liệu phục vụ suy luận đã được kiểm tra khớp 100% mã hash SHA-256 qua script `scripts/verify_production_artifacts.py`:

### CineAI (10 Mô hình Phim - MovieLens 100K):
- [x] `popularity.pkl` (317.0 KB) — `cf8904ed08605852...`
- [x] `user_cf.pkl` (5030.8 KB) — `f4d84a06d86b5fe4...`
- [x] `item_cf.pkl` (2344.5 KB) — `9312d159a31a6986...`
- [x] `knn_baseline.pkl` (2364.2 KB) — `8dbddab252e0acd5...`
- [x] `svd.pkl` (582.4 KB) — `c5b1eb5ade518e1e...`
- [x] `svdpp.pkl` (787.5 KB) — `b1fb718cae3a6a02...`
- [x] `nmf.pkl` (890.0 KB) — `00cff6362c019896...`
- [x] `content_based.pkl` (24971.7 KB) — `b99430c78d18e72b...`
- [x] `neural_cf.pkl` (1464.0 KB) — `9fb9c25aaf61eaf6...`
- [x] `hybrid.pkl` (30276.9 KB) — `db79d05aea99d341...`

### MusicAI (6 Mô hình Nhạc - Million Song Dataset):
- [x] `popularity.pkl` (1184.2 KB) — `ca60bea951a5d42a...`
- [x] `item_cf.pkl` (4033.8 KB) — `b4dde1c5096c553f...`
- [x] `user_cf.pkl` (3090.1 KB) — `274749f99ee7dbce...`
- [x] `ials.pkl` (1877.3 KB) — `913d8558bc833d7d...`
- [x] `bpr_mf.pkl` (1877.5 KB) — `cd2e3de247f4468f...`
- [x] `implicit_neumf.pkl` (2572.3 KB) — `59348528553c19e9...`

### Tập Dữ liệu Cốt lõi:
- [x] `data/processed/movies_clean.csv` (264.7 KB)
- [x] `data/processed/train.csv` (1937.3 KB)
- [x] `data/music/processed/songs.csv` (188.5 KB)
- [x] `data/music/processed/train.csv` (2278.6 KB)

---

## 3. BẢNG KIỂM TRA TRONG QUÁ TRÌNH TRIỂN KHAI (DEPLOYMENT CHECKLIST)

- [ ] **Render Web Service:** Đã tạo dịch vụ với runtime Docker và Dockerfile `./Dockerfile.backend`.
- [ ] **Render Region:** Đã chọn `Singapore` để tối ưu thời gian phản hồi.
- [ ] **Environment Variables trên Render:** Đã nhập đầy đủ `OPENBLAS_NUM_THREADS`, `MAX_CACHED_MODELS`, `CORS_ORIGINS`, `ARTIFACTS_DOWNLOAD_URL`.
- [ ] **Health Check Path trên Render:** Đã cấu hình `/api/v1/health`.
- [ ] **Vercel Frontend Environment Variables:** Đã nhập `VITE_API_BASE_URL` trỏ tới domain Render.
- [ ] **Vercel Redeploy:** Đã kích hoạt Redeploy để nhúng URL API mới vào mã nguồn Frontend.

---

## 4. TIÊU CHÍ NGHIỆM THU CUỐI CÙNG (GO / NO-GO CRITERIA)

| Điều kiện quyết định | Tiêu chí đánh giá | Trạng thái |
| :--- | :--- | :---: |
| **API Health** | Cả `/api/v1/health` và `/api/music/health` trả HTTP 200 trạng thái `healthy` | **GO** |
| **16 Mô hình Suy luận** | Tất cả 16 mô hình đều trả danh sách gợi ý hợp lệ, không lỗi 500 | **GO** |
| **Khử trùng lặp (Deduplication)** | Không gợi ý bài hát/phim người dùng đã tương tác trong tập Train | **GO** |
| **Bảo mật CORS** | Frontend Vercel gọi được API Backend mà không bị lỗi CORS chặn | **GO** |
| **Khả năng Phục hồi (SPA Routing)** | Người dùng tải lại trang bất kỳ không bị lỗi 404 | **GO** |
