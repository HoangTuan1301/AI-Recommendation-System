# BÁO CÁO KẾT QUẢ HUẤN LUYỆN 6 MÔ HÌNH MUSICAI
## PHASE M2: MODEL TRAINING, RESOURCE PROFILING & CHECKPOINTS

**Dự án:** Multi-Domain AI Recommendation System (MusicAI)  
**Ngày thực hiện:** 28/10/2026  
**Tổng thời gian thực thi chuỗi huấn luyện:** 64.00 giây  

---

## 1. BẢNG TỔNG HỢP HUẤN LUYỆN VÀ TÀI NGUYÊN (TRAINING SUMMARY)

| STT | Mô hình | Họ thuật toán | Thời gian huấn luyện (s) | Dung lượng RAM (MB) | Kích thước Checkpoint (KB) |
|:---:|---|---|:---:|:---:|:---:|
| 1 | **Popularity Baseline** | Heuristic / Baseline | 0.13s | 337.8 MB | 1,184.2 KB |
| 2 | **Item-Based CF** | Memory-Based Item CF | 0.37s | 371.6 MB | 4,033.8 KB |
| 3 | **User-Based CF** | Memory-Based User CF | 0.43s | 412.8 MB | 3,090.1 KB |
| 4 | **Implicit ALS** | Matrix Factorization (iALS) | 3.35s | 435.3 MB | 1,877.3 KB |
| 5 | **BPR-MF** | Pairwise Latent Factor (SGD) | 2.02s | 431.4 MB | 1,877.5 KB |
| 6 | **Implicit Neural CF** | Deep Learning (PyTorch NeuMF) | 56.7s | 528.8 MB | 2,572.3 KB |

---

## 2. CHI TIẾT CHECKPOINT ĐÃ TUẦN TỰ HÓA (`models/checkpoints/music/`)

- `popularity.pkl`: Vector trọng số độ phổ biến toàn cục.
- `item_cf.pkl`: Ma trận kề Cosine Top-50 láng giềng kèm tra cứu confidence.
- `user_cf.pkl`: Ma trận kề người dùng Top-50 kèm ma trận tương tác thưa.
- `ials.pkl`: Nhân tố người dùng $X \in \mathbb{R}^{2520 \times 32}$ và nhân tố bài hát $Y \in \mathbb{R}^{2706 \times 32}$.
- `bpr_mf.pkl`: Trọng số nhúng BPR $W \in \mathbb{R}^{2520 \times 32}$ và $H \in \mathbb{R}^{2706 \times 32}$.
- `implicit_neumf.pkl`: Toàn bộ cấu trúc và trọng số PyTorch GMF + MLP.
