# BÁO CÁO KIỂM SOÁT CHẤT LƯỢNG DỮ LIỆU (DATA QUALITY REPORT)
## PHASE M1: DATA SANITIZATION, SPLIT INTEGRITY & CHECKSUM AUDIT

**Tập dữ liệu:** MSD Taste Profile Song-Level Core Subset  
**Ngày kiểm toán:** 28/10/2026  
**Trạng thái kiểm tra:** 100% KIỂM TRA ĐẠT CHUẨN (ALL INTEGRITY CHECKS PASSED)  

---

## 1. BẢNG TỔNG KẾT RÀ SOÁT DỮ LIỆU (CLEANING SUMMARY)

| Tiêu chí kiểm tra chất lượng | Ràng buộc áp dụng | Kết quả kiểm tra | Trạng thái |
|---|---|---|---|
| **Giá trị khuyết thiếu User ID** | Không chấp nhận giá trị Null / NaN | 0 bản ghi lỗi | **ĐẠT** |
| **Giá trị khuyết thiếu Song ID** | Không chấp nhận giá trị Null / NaN | 0 bản ghi lỗi | **ĐẠT** |
| **Tính hợp lệ của lượt nghe** | `play_count >= 1` (Số nguyên dương) | 100% hợp lệ, loại bỏ các giá trị <= 0 | **ĐẠT** |
| **Trùng lặp cặp (user, song)** | Khóa chính duy nhất (user_id, song_id) | Hợp nhất bằng hàm Max play_count | **ĐẠT** |
| **Khớp nối Metadata** | 100% bài hát phải có trong `unique_tracks.txt` | 100% tìm thấy Title & Artist | **ĐẠT** |
| **Độ phủ danh mục Train (Warm-Start)** | All Val & Test items in Train catalog | 100% bài hát kiểm thử có trong tập Train | **ĐẠT** |
| **Tính rời rạc của các tập phân hoạch** | Train ∩ Val ∩ Test = ∅ | Giao cắt hoàn toàn bằng RỖNG | **ĐẠT** |

---

## 2. THỐNG KÊ CHI TIẾT TẬP PHÂN HOẠCH (SPLIT STATISTICS)

| Phân hoạch | Số lượng tương tác | Tỷ lệ thực tế | Số người dùng | Số bài hát |
|---|---|---|---|---|
| **Tập Huấn luyện (Train)** | **36,994** | 79.2% | 2,520 | 2,706 |
| **Tập Thẩm định (Validation)** | **4,844** | 10.4% | 2,520 | 1,974 |
| **Tập Kiểm thử (Test)** | **4,844** | 10.4% | 2,520 | 2,000 |
| **Tổng cộng** | **46,682** | **100.0%** | **2,520** | **2,706** |

---

## 3. MÃ BĂM TOÀN VẸN DỮ LIỆU (SHA-256 CHECKSUMS)

Các tệp được chứng thực mã băm SHA-256 bảo đảm tính bất biến:
- `interactions.csv`: `1e27cba0047d690763108823d68711c67394901a7a16b9c3365fe0586d3467a4`
- `train.csv`: `d687485ab61b7c1ec86e7eccdaac14d69a452fc44f661f3478ef0118ea0d3c43`
- `validation.csv`: `badcbe04038f4b1c944ef09cffa87e2db88473896399e3178dd1ff738f754c04`
- `test.csv`: `894cac7929dbbff404aeecd5819948dd7d47e48031bac9ea43889ab2ebc37b91`
- `songs.csv`: `53077d852a0cd47098f2b9ae941dc1ac745967e437e2bff8758e4cb794b6f9b6`

---

## 4. KẾT LUẬN

Tập dữ liệu hoàn toàn thỏa mãn mọi tiêu chí nghiêm ngặt của Hợp đồng Dữ liệu Phase M1 và sẵn sàng 100% để chuyển giao sang **Phase M2: Triển khai 6 Mô hình AI Gợi ý Âm nhạc**.
