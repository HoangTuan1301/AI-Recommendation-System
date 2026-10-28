# BÁO CÁO KIỂM TOÁN AN NINH VÀ BẢO MẬT (SECURITY & COMPLIANCE AUDIT)

**Hệ thống:** Multi-Domain AI Recommendation System (CineAI V2.0 + MusicAI)  
**Ngày kiểm toán:** 28/10/2026  
**Chuyên gia kiểm toán:** Senior Security Reviewer, DevOps & ML Systems Engineer  
**Phạm vi:** Mã nguồn, Container, Xử lý Serialization, API Endpoints, và Biến môi trường  

---

## 1. TỔNG HỢP KẾT QUẢ KIỂM TOÁN BẢO MẬT

| Lĩnh vực kiểm tra | Tiêu chuẩn đánh giá | Kết quả kiểm toán | Đánh giá |
|:---|:---|:---|:---:|
| **Quản lý Khóa & Secrets** | Không có API Key, Token, Password hardcoded trong git | Rà soát toàn bộ commit và working tree: **0 secret bị lộ**. `.env` được đưa vào `.gitignore`. | **PASS** |
| **Quyền hạn Container** | Không chạy container với quyền `root` | `Dockerfile.backend` chạy user `appuser` (UID 10001). `Dockerfile.frontend` chạy `nginx-unprivileged` (UID 101). | **PASS** |
| **An toàn Serialization (Pickle Safety)** | Không load pickle từ nguồn không xác thực bên ngoài | Tất cả checkpoint được nạp cục bộ từ đường dẫn mount được niêm phong `models/checkpoints`. Đã xác minh SHA-256 đối chiếu với manifest. | **PASS** |
| **Xác thực Đầu vào (Input Validation)** | Giới hạn `top_k`, kiểm tra độ dài chuỗi `user_id`, enum `model` | Sử dụng Pydantic và FastAPI Query validation: `top_k: ge=1, le=100`, `user_id: min_length=1`. | **PASS** |
| **Chính sách CORS** | Giới hạn danh sách domain được phép truy cập | Cấu hình tường minh danh sách `CORS_ORIGINS` trong `Settings`. Không sử dụng `allow_origins=["*"]` ở môi trường sản xuất. | **PASS** |
| **Rò rỉ Traceback (Information Disclosure)** | Không để lộ stack trace nội bộ khi gặp lỗi 500 | `global_exception_handler` trong FastAPI chặn toàn bộ exception và chỉ trả về thông báo lỗi tổng quát `detail: "An internal server error occurred."` | **PASS** |
| **Bảo mật Header Nginx** | Áp dụng các tiêu chuẩn an toàn web hiện đại | Đã thêm 4 headers: `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `X-XSS-Protection`, `Referrer-Policy: strict-origin-when-cross-origin`. | **PASS** |
| **Quyền riêng tư Dữ liệu (Privacy Boundary)** | Chỉ cung cấp dữ liệu huấn luyện công khai, không lộ test split | API `/api/music/users/{id}/history` được thiết kế chặt chẽ chỉ truy xuất từ `user_train_records` (nguồn `train.csv`), tuyệt đối không truy xuất từ `validation.csv` hay `test.csv`. | **PASS** |

---

## 2. CHI TIẾT CÁC PHÁT HIỆN VÀ BIỆN PHÁP KHẮC PHỤC

### 2.1. Phòng ngừa Lỗ hổng Tranh chấp Luồng OpenBLAS (Concurrency & Memory Safety)
- **Vấn đề:** Khi nhiều luồng truy cập đồng thời vào thư viện đại số tuyến tính OpenBLAS trên Windows/Linux, tiến trình có thể gặp lỗi `Memory allocation still failed after 10 retries`.
- **Khắc phục:** Đã cấu hình biến môi trường `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OMP_NUM_THREADS=1` ngay tại điểm khởi đầu của tiến trình (`backend/app/main.py`, `conftest.py`, `Dockerfile.backend`, và `docker-compose.yml`).

### 2.2. Kiểm soát Tấn công Từ chối Dịch vụ (DoS via Heavy Inference)
- **Vấn đề:** Người dùng có thể yêu cầu Top-K quá lớn (e.g. `top_k=1000000`) gây nghẽn CPU hoặc bộ nhớ.
- **Khắc phục:** Tham số `top_k` được giới hạn cứng tại tầng routing: `Query(default=10, ge=1, le=100)`. Bất kỳ yêu cầu nào vượt quá 100 sẽ bị FastAPI từ chối ngay lập tức với mã lỗi `422 Unprocessable Entity`.

---

## 3. KẾT LUẬN

Hệ thống tuân thủ nghiêm ngặt các tiêu chuẩn an toàn thông tin OWASP Top 10 và bảo mật hệ thống học máy. Không phát hiện bất kỳ lỗ hổng bảo mật nghiêm trọng (Critical/High) nào.
