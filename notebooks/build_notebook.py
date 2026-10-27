"""Generate 01_movielens_eda.ipynb notebook programmatically with nbformat."""

import nbformat as nbf

nb = nbf.v4.new_notebook()

# Metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (.venv)",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.11.9"
    }
}

cells = []

# Title & intro
cells.append(nbf.v4.new_markdown_cell("""# 01 - Phân Tích Khám Phá Dữ Liệu (EDA) - MovieLens 100K

**Dự án:** Hệ thống gợi ý phim / bài hát theo sở thích ứng dụng Trí tuệ nhân tạo (AI Recommendation System).  
**Học phần:** Trí tuệ nhân tạo (K5)  
**Dataset:** MovieLens 100K (GroupLens Research - University of Minnesota)  

---

### Mục tiêu phân tích:
1. Đánh giá tính toàn vẹn, phân bố và đặc tính thống kê của MovieLens 100K.
2. Đo lường độ thưa (sparsity) của ma trận tương tác User-Item.
3. Phân tích hiện tượng phân bố đuôi dài (Long-Tail Distribution) của phim và người dùng.
4. Đánh giá tác động của các đặc trưng dữ liệu đến việc thiết kế và lựa chọn **10 mô hình gợi ý** trong các giai đoạn tiếp theo.
"""))

# Cell 1: Setup
cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import module từ src
REPO_ROOT = Path("..").resolve()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from src.data_loader import load_movielens_100k, validate_movielens_data, GENRE_NAMES

# Cấu hình thẩm mỹ hiển thị biểu đồ
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
pd.set_option('display.max_columns', None)
print("Thư viện đã được tải thành công!")
"""))

# Cell 2: Load data
cells.append(nbf.v4.new_markdown_cell("## 1. Nạp và Xác Thực Dữ Liệu Gốc"))
cells.append(nbf.v4.new_code_cell("""ratings, movies, users = load_movielens_100k(data_dir=REPO_ROOT / "data" / "raw" / "ml-100k")
val_report = validate_movielens_data(ratings, movies, users)

print(f"Tổng số bản ghi đánh giá (ratings): {len(ratings):,}")
print(f"Tổng số người dùng (users):          {len(users):,}")
print(f"Tổng số bộ phim (movies):            {len(movies):,}")
print(f"Độ thưa ma trận tương tác:          {val_report['matrix_sparsity_percent']}%")
"""))

# Cell 3: Ratings distribution
cells.append(nbf.v4.new_markdown_cell("## 2. Phân Bố Điểm Đánh Giá (Ratings 1 đến 5 sao)"))
cells.append(nbf.v4.new_code_cell("""rating_counts = ratings['rating'].value_counts().sort_index()
rating_pct = (rating_counts / len(ratings)) * 100

dist_df = pd.DataFrame({
    'Số lượng': rating_counts,
    'Tỷ lệ (%)': rating_pct.round(2)
})
display(dist_df)

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(rating_counts.index, rating_counts.values, color='#2b5c8f', edgecolor='#1a365d', width=0.6)
for bar in bars:
    h = bar.get_height()
    pct = (h / len(ratings)) * 100
    ax.annotate(f"{h:,}\\n({pct:.1f}%)", xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title("Phân Bố Điểm Đánh Giá (1 đến 5 sao) - MovieLens 100K", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Mức điểm (Rating)", fontsize=10)
ax.set_ylabel("Số lượng đánh giá", fontsize=10)
ax.set_ylim(0, max(rating_counts.values) * 1.15)
plt.tight_layout()
plt.show()
"""))

# Cell 4: User Activity
cells.append(nbf.v4.new_markdown_cell("## 3. Hoạt Động Của Người Dùng (Ratings per User)"))
cells.append(nbf.v4.new_code_cell("""user_activity = ratings.groupby('user_id').size()
print(f"Số lượt đánh giá tối thiểu của 1 user: {user_activity.min()}")
print(f"Số lượt đánh giá trung bình của 1 user: {user_activity.mean():.2f}")
print(f"Trung vị số đánh giá:                  {user_activity.median()}")
print(f"Số lượt đánh giá tối đa của 1 user:     {user_activity.max()}")

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.histplot(user_activity, bins=40, kde=True, color='#1e88e5', edgecolor='#0d47a1', ax=ax)
ax.axvline(user_activity.median(), color='#d32f2f', linestyle='--', label=f"Trung vị: {int(user_activity.median())}")
ax.axvline(user_activity.mean(), color='#f57c00', linestyle=':', label=f"Trung bình: {user_activity.mean():.1f}")
ax.set_title("Phân Bố Số Lượng Đánh Giá Trên Mỗi Người Dùng", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Số lượt đánh giá của người dùng", fontsize=10)
ax.set_ylabel("Số lượng người dùng", fontsize=10)
ax.legend()
plt.tight_layout()
plt.show()
"""))

# Cell 5: Movie Popularity
cells.append(nbf.v4.new_markdown_cell("## 4. Hiện Tượng Đuôi Dài (Long-Tail Distribution)"))
cells.append(nbf.v4.new_code_cell("""movie_popularity = ratings.groupby('movie_id').size().sort_values(ascending=False).values

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(range(1, len(movie_popularity) + 1), movie_popularity, color='#00897b', linewidth=2)
ax.fill_between(range(1, len(movie_popularity) + 1), movie_popularity, color='#80cbc4', alpha=0.4)
ax.axvline(x=int(len(movie_popularity) * 0.2), color='#e53935', linestyle='--', label="Top 20% phim phổ biến nhất")
ax.set_title("Hiện Tượng Đuôi Dài Của Lượt Xem/Đánh Giá Phim (Log Scale)", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Thứ hạng phim (xếp theo độ phổ biến)", fontsize=10)
ax.set_ylabel("Số lượt đánh giá (Thang đo Log)", fontsize=10)
ax.set_yscale('log')
ax.legend()
plt.tight_layout()
plt.show()
"""))

# Cell 6: Top 10 Popular & Top 10 Highest Rated
cells.append(nbf.v4.new_markdown_cell("## 5. Top 10 Phim Phổ Biến Nhất & Top 10 Phim Điểm Cao Nhất"))
cells.append(nbf.v4.new_code_cell("""# Top 10 phổ biến nhất
top10_pop = (
    ratings.groupby('movie_id')
    .size()
    .reset_index(name='luot_danh_gia')
    .merge(movies[['movie_id', 'title']], on='movie_id')
    .sort_values('luot_danh_gia', ascending=False)
    .head(10)
)
print("=== TOP 10 PHIM CÓ NHIỀU ĐÁNH GIÁ NHẤT ===")
display(top10_pop[['movie_id', 'title', 'luot_danh_gia']])

# Top 10 điểm trung bình cao nhất (Ngưỡng tối thiểu 100 đánh giá để tránh thiên lệch mẫu nhỏ)
movie_stats = (
    ratings.groupby('movie_id')
    .agg(diem_tb=('rating', 'mean'), so_danh_gia=('rating', 'count'))
    .reset_index()
    .merge(movies[['movie_id', 'title']], on='movie_id')
)
top10_high = (
    movie_stats[movie_stats['so_danh_gia'] >= 100]
    .sort_values('diem_tb', ascending=False)
    .head(10)
)
print("\\n=== TOP 10 PHIM ĐIỂM TRUNG BÌNH CAO NHẤT (>= 100 ĐÁNH GIÁ) ===")
display(top10_high[['movie_id', 'title', 'diem_tb', 'so_danh_gia']])
"""))

# Cell 7: Genre Distribution
cells.append(nbf.v4.new_markdown_cell("## 6. Phân Bố Thể Loại Phim (19 Thể Loại)"))
cells.append(nbf.v4.new_code_cell("""genre_totals = movies[GENRE_NAMES].sum().sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(genre_totals.index[::-1], genre_totals.values[::-1], color='#5e35b1', edgecolor='#311b92')
for bar in bars:
    w = bar.get_width()
    pct = (w / len(movies)) * 100
    ax.annotate(f"{w:,} ({pct:.1f}%)", xy=(w, bar.get_y() + bar.get_height() / 2),
                xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9)
ax.set_title("Số Lượng Phim Theo Từng Thể Loại (MovieLens 100K)", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Số lượng phim", fontsize=10)
ax.set_xlim(0, max(genre_totals.values) * 1.18)
plt.tight_layout()
plt.show()
"""))

# Cell 8: Sparsity Analysis
cells.append(nbf.v4.new_markdown_cell("## 7. Phân Tích Độ Thưa Ma Trận (Matrix Sparsity)"))
cells.append(nbf.v4.new_code_cell("""total_cells = len(users) * len(movies)
observed_ratings = len(ratings)
unobserved_cells = total_cells - observed_ratings
sparsity = (unobserved_cells / total_cells) * 100

print(f"Tổng số ô trong ma trận User-Item ({len(users)} x {len(movies)}): {total_cells:,}")
print(f"Số ô đã có đánh giá (Observed):                       {observed_ratings:,} ({100 - sparsity:.2f}%)")
print(f"Số ô trống / chưa đánh giá (Unobserved):              {unobserved_cells:,} ({sparsity:.2f}%)")
"""))

# Cell 9: Temporal analysis
cells.append(nbf.v4.new_markdown_cell("## 8. Hoạt Động Đánh Giá Theo Dòng Thời Gian"))
cells.append(nbf.v4.new_code_cell("""ratings_time = ratings.copy()
ratings_time['date'] = pd.to_datetime(ratings_time['timestamp'], unit='s')
monthly_ratings = ratings_time.set_index('date').resample('ME').size()

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(monthly_ratings.index, monthly_ratings.values, color='#e53935', marker='o', linewidth=2)
ax.set_title("Số Lượng Lượt Đánh Giá Theo Tháng (09/1997 - 04/1998)", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Thời gian", fontsize=10)
ax.set_ylabel("Số lượng đánh giá", fontsize=10)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""))

# Cell 10: Conclusion & Recommendations
cells.append(nbf.v4.new_markdown_cell("""## 9. Nhận Xét Chuyên Sâu & Tác Động Đến Thuật Toán Gợi Ý

Qua quá trình thực nghiệm khám phá dữ liệu (EDA), chúng tôi rút ra các kết luận quan trọng cho bài toán gợi ý:

1. **Độ lệch phân bố điểm số (Rating Bias):**
   - Khoảng 55.4% tổng số lượt đánh giá rơi vào điểm 4 và 5 sao. Điểm 1 và 2 sao chỉ chiếm chưa đầy 18%.
   - *Tác động:* Cần thiết lập mô hình có thành phần điều chỉnh độ lệch bias (Baseline Estimates, KNNBaseline, SVD with Biases) để tránh dự đoán thiên lệch cho người dùng khó tính hoặc dễ tính.

2. **Hiện tượng đuôi dài (Long-Tail Distribution):**
   - Khoảng 20% phim phổ biến chiếm tới gần 80% tổng số lượt đánh giá (tiêu biểu như *Star Wars (1977)*, *Contact (1997)*, *Fargo (1996)*). Hàng trăm bộ phim khác chỉ có từ 1 đến 5 đánh giá.
   - *Tác động:* Các mô hình Baseline dựa trên độ phổ biến (Popularity Recommender) sẽ đạt độ chính xác khá tốt trên toàn bộ tập dữ liệu nhưng thiếu tính đa dạng (Diversity) và không gợi ý được các phim ngách (Serendipity). Các mô hình Lọc cộng tác (CF) cần áp dụng kỹ thuật co cụm hoặc lọc dựa trên nội dung (Content-Based) để giải quyết vấn đề Cold-Start cho item ít tương tác.

3. **Độ thưa ma trận cao (93.70% Sparsity):**
   - Mặc dù mỗi người dùng trong MovieLens 100K có tối thiểu 20 lượt đánh giá, mật độ ma trận vẫn chỉ đạt khoảng 6.30%.
   - *Tác động:* Biểu diễn dạng ma trận dày (Dense Matrix) sẽ gây lãng phí bộ nhớ khổng lồ. Bắt buộc phải sử dụng biểu diễn ma trận thưa (`scipy.sparse.csr_matrix`). Các thuật toán phân rã ma trận như **SVD**, **SVD++**, **NMF** và mạng nơ-ron **NCF** sẽ phát huy ưu thế vượt trội so với KNN truyền thống khi không gian chiều bị thiếu hụt nghiêm trọng.

4. **Định nghĩa tương tác tích cực (Positive Interaction) cho bài toán Top-N:**
   - Với thang điểm 1-5 sao, việc người dùng đánh giá 1-3 sao thường mang ý nghĩa không hài lòng hoặc trung tính.
   - *Quyết định thiết kế:* Định nghĩa **Rating $\\ge$ 4** là tương tác phù hợp (Relevant Item) để đánh giá các độ đo Top-N Ranking (Precision@K, Recall@K, NDCG@K, Hit Rate@K).
"""))

nb.cells = cells

# Save notebook
with open("notebooks/01_movielens_eda.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Saved notebooks/01_movielens_eda.ipynb successfully!")
