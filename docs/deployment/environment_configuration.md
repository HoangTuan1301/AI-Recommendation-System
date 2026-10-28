# CẤU HÌNH BIẾN MÔI TRƯỜNG VÀ AN TOÀN BẢO MẬT (ENVIRONMENT & SECURITY CONFIGURATION)

**Hệ thống:** CineAI V2.0 + MusicAI  
**Phiên bản:** 1.0  
**Giai đoạn:** Phase M6  

---

## 1. NGUYÊN TẮC BẢO MẬT VÀ QUẢN LÝ SECRETS

> [!CAUTION]
> Tuyệt đối không commit các khóa bí mật (API Keys, Token, Private Keys) vào Git repository.  
> Tệp `.env` thực tế nằm trong danh mục bị Git bỏ qua (`.gitignore`).  
> Dự án chỉ cung cấp tệp mẫu `.env.example` với các giá trị mặc định an toàn cho môi trường nội bộ.

---

## 2. DANH MỤC BIẾN MÔI TRƯỜNG (ENVIRONMENT VARIABLES)

### 2.1. Backend FastAPI Environment Variables
| Tên biến | Kiểu dữ liệu | Giá trị mặc định | Mô tả chi tiết |
|:---|:---:|:---:|:---|
| `OPENBLAS_NUM_THREADS` | Integer | `1` | **Bắt buộc = 1** để ngăn chặn lỗi cấp phát bộ nhớ và xung đột đa luồng trên Windows/Docker. |
| `MKL_NUM_THREADS` | Integer | `1` | Giới hạn luồng tính toán ma trận của Intel MKL. |
| `OMP_NUM_THREADS` | Integer | `1` | Giới hạn luồng tính toán OpenMP. |
| `PYTHONUNBUFFERED` | Integer | `1` | Đảm bảo log của Python được xuất trực tiếp ra stdout/stderr mà không bị giữ lại trong bộ đệm. |
| `CORS_ORIGINS` | JSON List | `["http://localhost:5173", "http://localhost:3000"]` | Danh sách các nguồn gốc hợp lệ được phép truy cập API thông qua CORS. |
| `PROJECT_NAME` | String | `CineAI + MusicAI Recommendation Platform` | Tên dự án hiển thị trong OpenAPI docs. |
| `VERSION` | String | `1.0.0` | Phiên bản hệ thống hiện tại. |
| `DEFAULT_MUSIC_MODEL` | String | `item_cf` | Mô hình mặc định cho MusicAI khi không chỉ định mô hình. |

### 2.2. Frontend Vite Environment Variables
| Tên biến | Kiểu dữ liệu | Giá trị mặc định | Mô tả chi tiết |
|:---|:---:|:---:|:---|
| `VITE_API_URL` | String | `/api` | Đường dẫn gốc của API (trong môi trường Docker / Dev được proxy tự động qua Nginx / Vite). |

---

## 3. TỆP MẪU `.env.example`

```env
# ==============================================================================
# Multi-Domain AI Recommendation System - Configuration Template
# ==============================================================================

# Scientific Computing Thread Allocation
OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1
OMP_NUM_THREADS=1
PYTHONUNBUFFERED=1

# CORS Allowed Origins (Comma-separated or JSON list)
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000","http://127.0.0.1:5173","http://127.0.0.1:3000"]

# Application Settings
PROJECT_NAME=Multi-Domain AI Recommendation System (CineAI + MusicAI)
VERSION=1.0.0
DEFAULT_MUSIC_MODEL=item_cf

# Port Configuration
BACKEND_PORT=8000
FRONTEND_PORT=3000
```
