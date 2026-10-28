# HỢP ĐỒNG API RESTFUL — MUSICAI PHASE M4
*(Official REST API Specification & Data Contract)*

**Dự án:** Multi-Domain AI Recommendation System (CineAI V2.0 & MusicAI)  
**Tài liệu:** `docs/musicai/phase_m4_api_contract.md`  
**Phiên bản API:** `1.0.0`  
**Tiền tố đường dẫn (Namespaces):** `/api/music` và `/api/v1/music`  
**Định dạng dữ liệu:** JSON (`application/json; charset=utf-8`)  

---

## 1. DANH SÁCH ENDPOINTS CHÍNH THỨC

| STT | Phương thức | Đường dẫn API | Chức năng chính |
|:---:|:---:|---|---|
| 1 | `GET` | `/api/music/health` | Kiểm tra trạng thái dịch vụ, quy mô catalog và mô hình khả dụng |
| 2 | `GET` | `/api/music/models` | Liệt kê danh sách 6 mô hình AI kèm thông tin thuật toán và trạng thái |
| 3 | `GET` | `/api/music/recommendations` | Tạo danh sách gợi ý bài hát Top-K theo người dùng và thuật toán |
| 4 | `GET` | `/api/music/songs/{song_id}` | Tra cứu chi tiết bài hát, nghệ sĩ và thống kê lượt nghe trong Train |
| 5 | `GET` | `/api/music/users/{user_id}/history` | Lấy danh sách lịch sử nghe nhạc của người dùng trong tập Train |

> [!NOTE]
> Mọi endpoint đều hỗ trợ đồng thời cả hai tiền tố: `/api/music/...` (đường dẫn trực tiếp) và `/api/v1/music/...` (đường dẫn chuẩn hoá theo phiên bản v1).

---

## 2. CHI TIẾT TỪNG ENDPOINT VÀ VÍ DỤ THỰC TẾ

### 2.1. GET /api/music/health
Trả về trạng thái hoạt động và các thông số tổng quan của miền âm nhạc.

#### Phản hồi thành công (200 OK):
```json
{
  "status": "healthy",
  "service": "MusicAI Song Recommendation Engine",
  "domain": "music",
  "catalog_songs": 2706,
  "total_users": 2520,
  "total_train_interactions": 36994,
  "available_models": 6,
  "loaded_models": ["item_cf", "popularity"],
  "default_model": "item_cf"
}
```

---

### 2.2. GET /api/music/models
Trả về danh mục 6 mô hình AI gợi ý bài hát đã được huấn luyện.

#### Phản hồi thành công (200 OK):
```json
{
  "total": 6,
  "default_model": "item_cf",
  "models": [
    {
      "id": "popularity",
      "name": "Popularity Baseline",
      "family": "Heuristic / Baseline",
      "description": "Gợi ý các bài hát có tổng lượt nghe nhiều nhất trên toàn hệ thống (không cá nhân hóa, dùng làm fallback).",
      "is_personalized": false,
      "is_default": false,
      "is_available": true,
      "is_loaded": true,
      "checkpoint_file": "popularity.pkl",
      "checkpoint_size_kb": 1184.2,
      "latency_profile": "< 1.0 ms"
    },
    {
      "id": "item_cf",
      "name": "Item-Based Collaborative Filtering",
      "family": "Neighborhood Collaborative Filtering",
      "description": "Gợi ý dựa trên ma trận đồng nghe Cosine Top-50 láng giềng. Đạt độ chính xác và độ phủ danh mục cao nhất.",
      "is_personalized": true,
      "is_default": true,
      "is_available": true,
      "is_loaded": true,
      "checkpoint_file": "item_cf.pkl",
      "checkpoint_size_kb": 4033.8,
      "latency_profile": "< 10.0 ms"
    },
    {
      "id": "user_cf",
      "name": "User-Based Collaborative Filtering",
      "family": "Neighborhood Collaborative Filtering",
      "description": "Tìm kiếm người dùng có gu âm nhạc tương đồng qua hệ số Cosine và tổng hợp các bài hát họ yêu thích.",
      "is_personalized": true,
      "is_default": false,
      "is_available": true,
      "is_loaded": false,
      "checkpoint_file": "user_cf.pkl",
      "checkpoint_size_kb": 3090.1,
      "latency_profile": "< 2.0 ms"
    },
    {
      "id": "ials",
      "name": "Implicit Alternating Least Squares (iALS)",
      "family": "Matrix Factorization",
      "description": "Phân rã ma trận tương tác ngầm định có trọng số tin cậy logarit sang không gian nhúng 32 chiều.",
      "is_personalized": true,
      "is_default": false,
      "is_available": true,
      "is_loaded": false,
      "checkpoint_file": "ials.pkl",
      "checkpoint_size_kb": 1877.3,
      "latency_profile": "< 5.0 ms"
    },
    {
      "id": "bpr_mf",
      "name": "Bayesian Personalized Ranking (BPR-MF)",
      "family": "Pairwise Ranking",
      "description": "Tối ưu hóa thứ hạng tương đối giữa bài hát đã nghe và chưa nghe qua hàm mất mát sigmoid BPR.",
      "is_personalized": true,
      "is_default": false,
      "is_available": true,
      "is_loaded": false,
      "checkpoint_file": "bpr_mf.pkl",
      "checkpoint_size_kb": 1877.5,
      "latency_profile": "< 6.0 ms"
    },
    {
      "id": "implicit_neural_cf",
      "name": "Implicit Neural Collaborative Filtering (NeuMF)",
      "family": "Deep Learning Hybrid",
      "description": "Mô hình học sâu kết hợp Generalized Matrix Factorization (GMF) và Multi-Layer Perceptron (MLP) phi tuyến.",
      "is_personalized": true,
      "is_default": false,
      "is_available": true,
      "is_loaded": false,
      "checkpoint_file": "implicit_neumf.pkl",
      "checkpoint_size_kb": 2572.3,
      "latency_profile": "< 4.0 ms"
    }
  ]
}
```

---

### 2.3. GET /api/music/recommendations
Tạo danh sách gợi ý bài hát Top-K cho người dùng mục tiêu.

#### Tham số truy vấn (Query Parameters):
- `user_id` (bắt buộc, string): Định danh người dùng.
- `model` (tùy chọn, string, mặc định: `"item_cf"`): Một trong 6 mã mô hình hợp lệ.
- `top_k` (tùy chọn, integer, mặc định: `10`, phạm vi $[1, 100]$): Số lượng bài hát mong muốn nhận được.

#### Ví dụ yêu cầu 1 (Người dùng đã có lịch sử nghe nhạc):
`GET /api/music/recommendations?user_id=0023d50b3c99c4352585408b1b21f3762655f434&model=item_cf&top_k=3`

#### Phản hồi (200 OK):
```json
{
  "user_id": "0023d50b3c99c4352585408b1b21f3762655f434",
  "model_requested": "item_cf",
  "model_used": "item_cf",
  "fallback_used": false,
  "fallback_reason": null,
  "top_k": 3,
  "count": 3,
  "inference_time_ms": 11.25,
  "recommendations": [
    {
      "rank": 1,
      "song_id": "SOTUKRT12A58A76AEB",
      "title": "Pimpin'",
      "artist_name": "Hollywood Undead",
      "track_id": "TRANGZB128F930B801",
      "score": 29.7086
    },
    {
      "rank": 2,
      "song_id": "SOWOAXK12A6310D81F",
      "title": "Undead",
      "artist_name": "Hollywood Undead",
      "track_id": "TRANHWS128F42750F8",
      "score": 28.1254
    },
    {
      "rank": 3,
      "song_id": "SOTVIDF12AB0182208",
      "title": "Young",
      "artist_name": "Hollywood Undead",
      "track_id": "TRBOVLP128F9307D5D",
      "score": 27.9412
    }
  ]
}
```

#### Ví dụ yêu cầu 2 (Khởi đầu lạnh — Người dùng hoàn toàn mới):
`GET /api/music/recommendations?user_id=guest_user_9999&top_k=2`

#### Phản hồi (200 OK — Fallback về Popularity Baseline):
```json
{
  "user_id": "guest_user_9999",
  "model_requested": "item_cf",
  "model_used": "popularity",
  "fallback_used": true,
  "fallback_reason": "User 'guest_user_9999' has no interaction history in training data. Fell back to non-personalized Popularity Baseline.",
  "top_k": 2,
  "count": 2,
  "inference_time_ms": 2.15,
  "recommendations": [
    {
      "rank": 1,
      "song_id": "SOBONKR12A58A7A7E0",
      "title": "You're The One",
      "artist_name": "Dwight Yoakam",
      "track_id": "TRAEHHJ12903CF492F",
      "score": 414.0
    },
    {
      "rank": 2,
      "song_id": "SOAUWYT12A81C206F1",
      "title": "Undo",
      "artist_name": "Björk",
      "track_id": "TRAGXRA128F427CF18",
      "score": 382.0
    }
  ]
}
```

---

### 2.4. GET /api/music/songs/{song_id}
Truy xuất thông tin chi tiết của một bài hát trong danh mục 2.706 bài hát đã nghiệm thu.

#### Ví dụ: `GET /api/music/songs/SOQMFQX12AB0189E9A`
#### Phản hồi (200 OK):
```json
{
  "song_id": "SOQMFQX12AB0189E9A",
  "track_id": "TRMMUMK12903CA7B75",
  "title": "Here I Am",
  "artist_name": "SOJA",
  "total_play_count": 15,
  "unique_listeners": 7
}
```

#### Phản hồi khi không tìm thấy (404 Not Found):
```json
{
  "detail": "Song with ID 'UNKNOWN_SONG_ID' not found in catalog."
}
```

---

### 2.5. GET /api/music/users/{user_id}/history
Tra cứu lịch sử nghe nhạc của người dùng trong tập Train (bảo đảm không rò rỉ dữ liệu Validation hoặc Test).

#### Ví dụ: `GET /api/music/users/0023d50b3c99c4352585408b1b21f3762655f434/history`
#### Phản hồi (200 OK):
```json
{
  "user_id": "0023d50b3c99c4352585408b1b21f3762655f434",
  "total_plays": 16,
  "total_songs": 4,
  "history": [
    {
      "song_id": "SOGIWFN12AB01884AD",
      "title": "Whatever You Like",
      "artist_name": "T.I.",
      "track_id": "TRMMFUB128F9329711",
      "play_count": 6
    },
    {
      "song_id": "SODJVTF12A8AE46DFD",
      "title": "Poker Face",
      "artist_name": "Lady GaGa",
      "track_id": "TRMMYUS128F93025A4",
      "play_count": 5
    },
    {
      "song_id": "SOTFTTK12A8C12FC79",
      "title": "Just Dance",
      "artist_name": "Lady GaGa / Colby O'Donis",
      "track_id": "TRMMDDE128F92F4A0A",
      "play_count": 4
    },
    {
      "song_id": "SOZSFWF12A6D4F93D5",
      "title": "Stronger",
      "artist_name": "Kanye West",
      "track_id": "TRMMSLN128F426399E",
      "play_count": 1
    }
  ]
}
```

---

## 3. BẢNG MÃ LỖI VÀ XỬ LÝ NGOẠI LỆ

| Mã HTTP | Ý nghĩa | Điều kiện kích hoạt |
|:---:|---|---|
| **200 OK** | Thành công | Yêu cầu hợp lệ, trả về kết quả đúng cấu trúc Pydantic |
| **400 Bad Request** | Lỗi tham số truy vấn | `top_k` ngoài khoảng $[1, 100]$, `user_id` rỗng, hoặc `model` không hợp lệ |
| **404 Not Found** | Không tìm thấy tài nguyên | `song_id` không tồn tại trong danh mục hoặc `user_id` không có lịch sử |
| **422 Unprocessable** | Sai định dạng kiểu dữ liệu | Tham số truy vấn thiếu trường bắt buộc hoặc sai kiểu (tự động từ FastAPI) |
| **503 Service Unavailable** | Dịch vụ gián đoạn | Tệp checkpoint bị thiếu hoặc lỗi không thể nạp vào bộ nhớ |
| **500 Internal Error** | Lỗi nội bộ không xác định | Được bắt tại Global Exception Handler, không làm lộ stacktrace |
