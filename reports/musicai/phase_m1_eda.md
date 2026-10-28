# BÁO CÁO PHÂN TÍCH KHÁM PHÁ DỮ LIỆU ÂM NHẠC (EDA REPORT)
## PHASE M1: SONG-LEVEL DATASET EXPLORATION & STATISTICAL PROFILING

**Tập dữ liệu:** Million Song Dataset (MSD) – Taste Profile Subset  
**Thực thể gợi ý chính:** Bài hát (`song_id`) kèm Metadata (`title`, `artist_name`, `track_id`)  
**Ngày thực hiện:** 28/10/2026  

---

## 1. TỔNG QUAN TẬP DỮ LIỆU ĐÃ TRÍCH XUẤT (SUBSET SUMMARY)

| Chỉ số thống kê | Giá trị định lượng | Ghi chú kỹ thuật |
|---|---|---|
| **Tổng số người dùng (|U|)** | **2,520** | Người dùng thực tế có hoạt động nghe tích cực |
| **Tổng số bài hát (|I|)** | **2,706** | Bài hát có metadata đầy đủ 100% |
| **Tổng số tương tác (|R|)** | **46,682** | Cặp (user_id, song_id, play_count) hợp lệ |
| **Độ thưa thớt (Sparsity)** | **99.32%** | Ma trận tương tác thưa thớt chuẩn RecSys |
| **Độ đầy (Density)** | **0.68%** | Mật độ tương tác kết nối người dùng - bài hát |
| **Bình quân bài hát / người dùng** | **18.52** | Trung vị: 14 bài/người dùng |
| **Bình quân người nghe / bài hát** | **17.25** | Trung vị: 11 người/bài hát |

---

## 2. PHÂN PHỐI HOẠT ĐỘNG NGƯỜI DÙNG (USER ACTIVITY DISTRIBUTION)

| Khoảng số lượng bài hát đã nghe | Số lượng người dùng | Tỷ lệ (%) |
|---|---|---|
| 1-10 bài hát | 818 | 32.5% |
| 11-20 bài hát | 931 | 36.9% |
| 21-50 bài hát | 677 | 26.9% |
| 51-100 bài hát | 87 | 3.5% |
| 101-200 bài hát | 7 | 0.3% |
| 201-500 bài hát | 0 | 0.0% |
| 501+ bài hát | 0 | 0.0% |

---

## 3. PHÂN BỐ SỐ LƯỢT NGHE (PLAY COUNT DISTRIBUTION)

Khác với điểm số phim (1–5 sao), số lượt nghe phản ánh thói quen tiêu dùng lặp lại (heavy tail distribution):
- **Giá trị nhỏ nhất (Min):** 1 lượt nghe
- **Phân vị 25% (P25):** 1.0 lượt nghe
- **Trung vị (Median):** 1.0 lượt nghe
- **Phân vị 75% (P75):** 3.0 lượt nghe
- **Phân vị 95% (P95):** 12.0 lượt nghe
- **Giá trị lớn nhất (Max):** 305 lượt nghe

> **Quy tắc chuyển đổi đề xuất cho Phase M2:**  
> Áp dụng hàm logarit làm mượt: $c_{ui} = 1 + \alpha \log(1 + r_{ui})$ với $\alpha = 40.0$ để tránh hiện tượng các bài hát nghe hàng trăm lần chi phối toàn bộ gradient của mô hình.

---

## 4. TOP 15 BÀI HÁT ĐƯỢC NGHE NHIỀU NHẤT (POPULARITY RANKINGS)

### 4.1. Top bài hát theo số lượng người nghe độc nhất (Distinct Listeners)

| Thứ hạng | Tên bài hát | Nghệ sĩ biểu diễn | Số người nghe độc nhất |
|---|---|---|---|
| 1 | Sehr kosmisch | Harmonia | 338 |
| 2 | Dog Days Are Over (Radio Edit) | Florence + The Machine | 269 |
| 3 | Undo | Björk | 257 |
| 4 | You're The One | Dwight Yoakam | 255 |
| 5 | Secrets | OneRepublic | 251 |
| 6 | Revelry | Kings Of Leon | 229 |
| 7 | Fireflies | Charttraxx Karaoke | 218 |
| 8 | Horn Concerto No. 4 in E flat K495: II. Romance (Andante cantabile) | Barry Tuckwell/Academy of St Martin-in-the-Fields/Sir Neville Marriner | 199 |
| 9 | Hey_ Soul Sister | Train | 194 |
| 10 | Use Somebody | Kings Of Leon | 176 |
| 11 | Tive Sim | Cartola | 173 |
| 12 | Drop The World | Lil Wayne / Eminem | 170 |
| 13 | OMG | Usher featuring will.i.am | 169 |
| 14 | Catch You Baby (Steve Pitron & Max Sanna Radio Edit) | Lonnie Gordon | 145 |
| 15 | Clocks | Coldplay | 141 |

### 4.2. Top bài hát theo tổng số lượt nghe tích lũy (Total Play Count)

| Thứ hạng | Tên bài hát | Nghệ sĩ biểu diễn | Tổng lượt nghe tích lũy |
|---|---|---|---|
| 1 | You're The One | Dwight Yoakam | 2,549 |
| 2 | Undo | Björk | 1,962 |
| 3 | Sehr kosmisch | Harmonia | 1,501 |
| 4 | Revelry | Kings Of Leon | 1,432 |
| 5 | Horn Concerto No. 4 in E flat K495: II. Romance (Andante cantabile) | Barry Tuckwell/Academy of St Martin-in-the-Fields/Sir Neville Marriner | 1,412 |
| 6 | Dog Days Are Over (Radio Edit) | Florence + The Machine | 1,108 |
| 7 | Secrets | OneRepublic | 1,107 |
| 8 | Canada | Five Iron Frenzy | 1,057 |
| 9 | Représente | Alliance Ethnik | 854 |
| 10 | Fireflies | Charttraxx Karaoke | 816 |
| 11 | The Gift | Angels and Airwaves | 784 |
| 12 | Catch You Baby (Steve Pitron & Max Sanna Radio Edit) | Lonnie Gordon | 761 |
| 13 | Hey_ Soul Sister | Train | 735 |
| 14 | Tive Sim | Cartola | 689 |
| 15 | Marry Me | Train | 644 |

---

## 5. KẾT LUẬN VÀ Ý NGHĨA CHO GIAI ĐOẠN HUẤN LUYỆN (PHASE M2)

1. Tập dữ liệu trích xuất bảo đảm tính kết nối cao (dense bipartite subgraph), giải quyết tốt hiện tượng Cold-Start cục bộ mà vẫn phản ánh đúng bản chất đuôi dài (Long-tail) của dữ liệu âm nhạc thế giới thực.
2. Metadata đạt tỷ lệ bao phủ 100%, bảo đảm giao diện và kết quả gợi ý luôn hiển thị rõ ràng tên bài hát và nghệ sĩ.
