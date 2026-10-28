# BÁO CÁO ĐO LƯỜNG HIỆU NĂNG VÀ TÀI NGUYÊN BACKEND — MUSICAI PHASE M4
*(Phase M4 Performance, Resource Profiling & Stress Testing Report)*

**Dự án:** Multi-Domain AI Recommendation System (CineAI V2.0 & MusicAI)  
**Tài liệu:** `reports/musicai/phase_m4_performance.md`  
**Giai đoạn:** Phase M4 — Multi-Domain FastAPI Backend  
**Ngày thực hiện:** 28/10/2026  
**Chuyên gia thực hiện:** Senior ML Systems Engineer & Performance Architect  
**Môi trường thử nghiệm:** Windows 11, Python 3.11.9, CPU x86_64, Uvicorn/FastAPI ASGI Test Runner  

---

## 1. TỔNG QUAN KẾT QUẢ ĐO LƯỜNG HIỆU NĂNG

Mục tiêu cốt lõi của Phase M4 là xác thực dịch vụ API thực tế (HTTP End-to-End) có đạt ngưỡng phản hồi thời gian thực ($< 50 \text{ ms}$) hay không. Chúng tôi thực hiện phân tách rõ ràng giữa **Thời gian suy luận mô hình ngoại tuyến (Offline Model Inference)** và **Độ trễ API thực tế (HTTP End-to-End Latency)** bao gồm: tiếp nhận HTTP request, phân tích JSON query, xác thực Pydantic, tra cứu metadata catalog, khử trùng lặp và tuần tự hóa phản hồi JSON.

### Bảng tóm tắt các chỉ số hiệu năng chính:

| Tiêu chí đo lường | Giá trị thực tế | Ngưỡng mục tiêu (SLA) | Trạng thái |
|---|:---:|:---:|:---:|
| **Thời gian khởi động Backend (Startup Time)** | **5.73 s** | $< 15.0 \text{ s}$ | **ĐẠT (Xuất sắc)** |
| **Dung lượng RAM sau khởi động (Catalog Loaded)** | **364.6 MB** | $< 500.0 \text{ MB}$ | **ĐẠT (Rất tối ưu)** |
| **Dung lượng RAM khi nạp đủ 6 mô hình** | **441.2 MB** | $< 800.0 \text{ MB}$ | **ĐẠT (An toàn cho laptop)** |
| **Độ trễ HTTP End-to-End trung bình (Item-CF)** | **18.15 ms** | $< 50.0 \text{ ms}$ | **ĐẠT (Vượt xa kỳ vọng)** |
| **Độ trễ HTTP P95 (Item-CF)** | **22.17 ms** | $< 50.0 \text{ ms}$ | **ĐẠT** |
| **Độ trễ HTTP P99 (Item-CF)** | **24.32 ms** | $< 70.0 \text{ ms}$ | **ĐẠT** |
| **Thông lượng đơn nhân (Single-Core Throughput)**| **55.1 req/s** | $> 20.0 \text{ req/s}$ | **ĐẠT** |
| **Tỷ lệ lỗi (Error Rate)** | **0.00% (0/100 req)** | $0.00\%$ | **100% SUCCESS** |

---

## 2. PHÂN TÁCH THỜI GIAN VÀ BỘ NHỚ THEO TỪNG MÔ HÌNH

### 2.1. Thời gian nạp Checkpoint từ ổ đĩa (Lazy Loading Latency)
Cơ chế Lazy Loading của `MusicModelRegistry` cho phép backend khởi động tức thì mà không cần nạp trước toàn bộ checkpoint. Thời gian nạp lần đầu (cold load) cho từng mô hình:

| Mã mô hình | Tên mô hình AI | Kích thước Checkpoint | Thời gian nạp (ms) | Dung lượng RAM tăng thêm (MB) |
|---|---|:---:|:---:|:---:|
| `popularity` | Popularity Baseline | 1.18 MB | **100.3 ms** | +5.8 MB |
| `item_cf` | Item-Based CF (Default) | 4.03 MB | **697.8 ms** | +31.4 MB |
| `user_cf` | User-Based CF | 3.09 MB | **316.0 ms** | +14.9 MB |
| `ials` | Implicit ALS | 1.88 MB | **121.8 ms** | +6.9 MB |
| `bpr_mf` | BPR-MF | 1.88 MB | **291.7 ms** | +7.0 MB |
| `implicit_neural_cf` | Implicit Neural CF | 2.57 MB | **119.2 ms** | +10.6 MB |

> [!NOTE]
> Thời gian nạp checkpoint chỉ xuất hiện một lần duy nhất tại request đầu tiên của mô hình đó (Warmup). Từ request thứ hai trở đi, mô hình được tái sử dụng trực tiếp từ bộ nhớ đệm RAM (`_loaded_models`) với thời gian lấy véc-tơ tham chiếu gần như bằng 0 ms ($O(1)$ dictionary lookup).

---

## 3. PHÂN RÃ ĐỘ TRỄ YÊU CẦU GỢI Ý (LATENCY DECOMPOSITION)

Đối với mô hình mặc định **Item-Based CF** với $K=10$ bài hát gợi ý cho người dùng đang hoạt động:

```
┌────────────────────────────────────────────────────────────────────────┐
│  HTTP End-to-End Latency: 18.15 ms                                     │
├───────────────────────────────────┬────────────────────────────────────┤
│  Model Inference: 9.56 ms (52.7%) │  Web/Service Overhead: 8.59 ms (47.3%)
│  - Ma trận Cosine Top-50 láng giềng│  - HTTP Routing & Query Parsing   │
│  - Loại bỏ bài hát đã nghe (Train) │  - Ánh xạ Metadata (Title, Artist)│
│  - Sắp xếp điểm số giảm dần       │  - Khử trùng lặp & Format JSON    │
└───────────────────────────────────┴────────────────────────────────────┘
```

1. **Pure Model Inference:** Chiếm **9.56 ms**. Đây là thời gian nhân ma trận độ tương đồng thưa và sắp xếp ứng viên trong không gian 2.706 bài hát.
2. **Service & Serialization Overhead:** Chiếm **8.59 ms**, bao gồm:
   - Tra cứu từ điển metadata bài hát ($O(1)$).
   - Xác thực schema Pydantic.
   - Ghi log hoạt động an toàn.
   - Mã hóa JSON response gửi về client.

---

## 4. KẾT QUẢ KIỂM THỬ TẢI YÊU CẦU LIÊN TIẾP (STRESS TEST)

Thực nghiệm gửi liên tiếp 100 requests tới endpoint `/api/music/recommendations?user_id=...&model=item_cf&top_k=10`:

```
Độ trễ phân vị (Percentiles):
  Min   : 16.20 ms
  P50   : 17.83 ms
  P75   : 19.05 ms
  P95   : 22.17 ms
  P99   : 24.32 ms
  Max   : 25.10 ms
```

### Nhận xét:
- **Độ ổn định cao:** Độ lệch chuẩn giữa các lần gọi cực thấp ($\sigma \approx 1.8 \text{ ms}$). Toàn bộ 100 requests đều hoàn thành dưới ngưỡng 26 ms.
- **Không xuất hiện hiện tượng rò rỉ bộ nhớ (Memory Leak):** Bộ nhớ RSS của tiến trình uvicorn duy trì phẳng (flat line) ở mức 441 MB trong suốt quá trình stress test.

---

## 5. HIỆU NĂNG CỦA CÁC ENDPOINTS PHỤ TRỢ

| Endpoint | Chức năng | Độ trễ trung bình | Ghi chú vận hành |
|---|---|:---:|---|
| `GET /api/music/health` | Kiểm tra tình trạng sẵn sàng | **2.10 ms** | Trả về thống kê catalog và trạng thái mô hình |
| `GET /api/music/models` | Liệt kê 6 mô hình | **2.85 ms** | Quét trạng thái checkpoint và RAM cache |
| `GET /api/music/songs/{id}` | Chi tiết bài hát | **6.49 ms** | Tra cứu tiêu đề, nghệ sĩ, lượt nghe và người nghe |
| `GET /api/music/users/{id}/history`| Lịch sử nghe nhạc Train | **6.72 ms** | Lấy danh sách bài hát người dùng đã nghe trong Train |
| `GET /api/music/recommendations` | Gợi ý người dùng mới (Cold-Start) | **7.50 ms** | Tự động kích hoạt Popularity Baseline fallback |

---

## 6. KẾT LUẬN VỀ HIỆU NĂNG
- Hệ thống backend MusicAI đã đạt hiệu năng vận hành xuất sắc: Thời gian phản hồi thực tế của API trung bình đạt **~18 ms**, mức tiêu thụ RAM tối đa **~441 MB**.
- Hoàn toàn đủ điều kiện kỹ thuật để phục vụ người dùng tương tác trực tiếp trên giao diện Web mà không gây giật, lag hoặc timeout.
