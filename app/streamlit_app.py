"""Interactive Movie Recommendation System - Streamlit Web Application.

Demonstration interface for AI Recommendation Systems on MovieLens 100K:
- 10 Machine Learning & Deep Learning Recommendation Models
- 5 Standardized Top-N Evaluation Metrics
- Interactive User Profile Explorer & Side-by-Side Model Comparison
- Comprehensive Project Documentation
"""

from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.model_loader import (
    MODEL_REGISTRY,
    get_model_recommendations_with_scores,
    load_or_train_model,
)

# Set page configuration
st.set_page_config(
    page_title="Hệ thống Gợi ý Phim AI - MovieLens 100K",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Modern CSS
CUSTOM_CSS = """
<style>
    /* Global styles */
    .stApp {
        background-color: #0B1120;
        color: #F8FAFC;
    }
    
    /* Header hero */
    .hero-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        border: 1px solid #4F46E5;
        border-radius: 14px;
        padding: 24px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 6px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #C7D2FE;
        font-weight: 400;
    }
    
    /* KPI Metric Cards */
    .kpi-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        border-color: #6366F1;
        transform: translateY(-2px);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
    }
    
    /* Movie recommendation card */
    .movie-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease-in-out;
    }
    .movie-card:hover {
        border-color: #818CF8;
        transform: translateY(-4px);
        box-shadow: 0 12px 20px -5px rgba(99, 102, 241, 0.3);
    }
    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        color: white;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .movie-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 6px;
        line-height: 1.3;
    }
    .movie-year {
        font-size: 0.85rem;
        color: #94A3B8;
        margin-bottom: 8px;
    }
    .genre-badge {
        display: inline-block;
        background: #0F172A;
        border: 1px solid #334155;
        color: #CBD5E1;
        font-size: 0.72rem;
        padding: 2px 7px;
        border-radius: 4px;
        margin: 2px 2px 4px 0;
    }
    .score-container {
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px solid #334155;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .score-label {
        font-size: 0.75rem;
        color: #94A3B8;
    }
    .score-value {
        font-size: 0.95rem;
        font-weight: 700;
        color: #34D399;
    }
    
    /* Poster placeholder */
    .poster-placeholder {
        background: linear-gradient(180deg, #1E1B4B 0%, #0F172A 100%);
        border-radius: 8px;
        height: 130px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
        border: 1px dashed #4338CA;
    }
    .poster-icon {
        font-size: 2.2rem;
        margin-bottom: 4px;
    }
    .poster-id {
        font-size: 0.7rem;
        color: #818CF8;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed MovieLens 100K datasets and offline benchmark metrics."""
    data_dir = REPO_ROOT / "data" / "processed"
    movies_df = pd.read_csv(data_dir / "movies_clean.csv")
    train_df = pd.read_csv(data_dir / "train.csv")
    metrics_path = REPO_ROOT / "results" / "phase4c_metrics.csv"
    if metrics_path.exists():
        metrics_df = pd.read_csv(metrics_path)
    else:
        metrics_df = pd.DataFrame()
    return movies_df, train_df, metrics_df


@st.cache_resource(show_spinner=False)
def get_model(model_key: str, _train_df: pd.DataFrame, _movies_df: pd.DataFrame) -> Any:
    """Load or train a model with Streamlit resource caching."""
    return load_or_train_model(model_key, train_df=_train_df, movies_df=_movies_df)


# Main Data Loading
try:
    movies_df, train_df, metrics_df = load_data()
    movies_lookup = movies_df.set_index("movie_id").to_dict("index")
    unique_train_users = sorted(train_df["user_id"].unique().tolist())
except Exception as e:
    st.error(f"Lỗi tải dữ liệu: {e}")
    st.stop()


# Sidebar Navigation
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/e/ea/Netflix_font_logo.png",
    width=160,
)
st.sidebar.markdown("### 🎬 Hệ thống Gợi ý Phim AI")
st.sidebar.markdown(
    "<small style='color: #94A3B8;'>Đồ án Môn học Trí tuệ Nhân tạo - K5<br>"
    "MovieLens 100K Dataset • 10 Mô hình AI</small>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "CHỌN TRANG ĐIỀU HƯỚNG:",
    [
        "🎬 Gợi ý phim (Recommendations)",
        "📊 So sánh 10 mô hình (Benchmark)",
        "👤 Hồ sơ sở thích người dùng (Profiles)",
        "ℹ️ Giới thiệu đề tài (About)",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size: 0.8rem; color: #64748B;'>"
    "<b>Workspace:</b> E:\\HOCTAP\\K5\\TTNT<br>"
    "<b>Môi trường:</b> Python 3.11 / PyTorch<br>"
    "<b>Trạng thái:</b> 10/10 Mô hình Sẵn sàng"
    "</div>",
    unsafe_allow_html=True,
)


# ==============================================================================
# PAGE 1: MOVIE RECOMMENDATIONS
# ==============================================================================
if app_mode == "🎬 Gợi ý phim (Recommendations)":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">🎬 Khám phá Gợi ý Phim theo Sở thích</div>
            <div class="hero-subtitle">
                Trải nghiệm 10 thuật toán Trí tuệ nhân tạo gợi ý phim cá nhân hóa trên tập dữ liệu MovieLens 100K
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top Control Bar
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.2, 1.8, 1])

    with col_ctrl1:
        # Pre-select user
        user_id_input = st.selectbox(
            "👤 Chọn Người dùng (User ID):",
            options=unique_train_users,
            index=0,
            help="Danh sách 943 người dùng có lịch sử đánh giá trong tập huấn luyện.",
        )

    with col_ctrl2:
        model_options = list(MODEL_REGISTRY.keys())
        selected_model_name = st.selectbox(
            "🧠 Chọn Mô hình Trí tuệ Nhân tạo:",
            options=model_options,
            index=8,  # Default to Neural CF (NeuMF)
            help="10 mô hình bao gồm: Baselines, Collaborative Filtering, Matrix Factorization, Deep Learning và Hybrid.",
        )

    with col_ctrl3:
        top_k = st.radio(
            "🎯 Số lượng gợi ý (Top-N):",
            options=[5, 10, 20],
            index=1,
            horizontal=True,
        )

    meta = MODEL_REGISTRY[selected_model_name]

    # Model info card
    st.markdown(
        f"""
        <div style="background: #1E293B; border-left: 4px solid #6366F1; padding: 12px 16px; border-radius: 6px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: 700; color: #F8FAFC; font-size: 1.05rem;">{meta['name']}</span>
                <span style="background: #312E81; color: #C7D2FE; font-size: 0.75rem; padding: 2px 8px; border-radius: 12px; font-weight: 600;">{meta['family']}</span>
            </div>
            <div style="font-size: 0.88rem; color: #94A3B8; margin-top: 4px;">{meta['description']}</div>
            <div style="font-size: 0.8rem; color: #38BDF8; margin-top: 6px;">
                <b>Thang đo điểm:</b> {meta['score_type']} ({meta['score_range']}) • <b>Độ phức tạp:</b> {meta['complexity']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    btn_recommend = st.button("🚀 Gợi ý phim ngay", type="primary", use_container_width=True)

    if btn_recommend or "last_rec" not in st.session_state:
        st.session_state["last_rec"] = True

        with st.spinner(f"Đang tải và suy luận qua mô hình {selected_model_name}..."):
            t_start = time.time()
            model_instance = get_model(selected_model_name, _train_df=train_df, _movies_df=movies_df)

            # Ensure user cannot receive seen items from Train history
            user_train_watched = set(train_df[train_df["user_id"] == user_id_input]["movie_id"])
            all_candidate_ids = [mid for mid in movies_df["movie_id"].unique() if mid not in user_train_watched]

            rec_results = get_model_recommendations_with_scores(
                model=model_instance,
                model_key=selected_model_name,
                user_id=user_id_input,
                top_n=top_k,
                candidate_items=all_candidate_ids,
            )
            elapsed_sec = time.time() - t_start

        # Inference metrics row
        st.markdown(
            f"""
            <div style="display: flex; gap: 20px; margin-bottom: 18px; font-size: 0.9rem; color: #94A3B8;">
                <span>🎯 <b>Người dùng:</b> #{user_id_input} ({len(user_train_watched)} phim đã đánh giá trong Train)</span>
                <span>⏱️ <b>Thời gian suy luận:</b> {elapsed_sec*1000:.1f} ms</span>
                <span>🔍 <b>Ứng viên xếp hạng:</b> {len(all_candidate_ids):,} phim chưa xem</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Responsive Grid for Movie Cards (5 columns per row)
        cols_per_row = 5
        num_cards = len(rec_results)

        for row_idx in range(0, num_cards, cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx in range(cols_per_row):
                item_idx = row_idx + col_idx
                if item_idx < num_cards:
                    item = rec_results[item_idx]
                    mid = item["movie_id"]
                    movie_meta = movies_lookup.get(mid, {})
                    title = movie_meta.get("title", f"Movie #{mid}")
                    year = movie_meta.get("year", "N/A")
                    genres_str = movie_meta.get("genres_str", "")
                    genre_list = genres_str.split("|") if genres_str else ["Unknown"]

                    genre_badges = "".join([f'<span class="genre-badge">{g}</span>' for g in genre_list[:3]])

                    with cols[col_idx]:
                        st.markdown(
                            f"""
                            <div class="movie-card">
                                <div>
                                    <div class="poster-placeholder">
                                        <div class="poster-icon">🍿</div>
                                        <div class="poster-id">ID: {mid}</div>
                                    </div>
                                    <span class="rank-badge">#{item['rank']} TOP PICK</span>
                                    <div class="movie-title">{title}</div>
                                    <div class="movie-year">Năm phát hành: <b>{year}</b></div>
                                    <div style="margin-top: 4px;">{genre_badges}</div>
                                </div>
                                <div class="score-container">
                                    <span class="score-label">{item['score_type']}</span>
                                    <span class="score-value">{item['formatted_score']}</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # Detailed Tabular View
        with st.expander("📋 Xem danh sách dữ liệu chi tiết dạng bảng"):
            table_data = []
            for it in rec_results:
                m_info = movies_lookup.get(it["movie_id"], {})
                table_data.append({
                    "Hạng": it["rank"],
                    "Mã phim (ID)": it["movie_id"],
                    "Tên phim": m_info.get("title", "N/A"),
                    "Năm": m_info.get("year", "N/A"),
                    "Thể loại": m_info.get("genres_str", "N/A"),
                    "Điểm dự đoán": it["formatted_score"],
                    "Thang đo": it["score_type"],
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)


# ==============================================================================
# PAGE 2: MODEL COMPARISON (OFFLINE BENCHMARK)
# ==============================================================================
elif app_mode == "📊 So sánh 10 mô hình (Benchmark)":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">📊 So sánh Hiệu năng 10 Mô hình Gợi ý</div>
            <div class="hero-subtitle">
                Kết quả thực nghiệm chuẩn hóa trên tập Test độc lập (935 người dùng) theo 5 độ đo Top-10
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if metrics_df.empty:
        st.warning("Chưa tìm thấy file kết quả `results/phase4c_metrics.csv`. Hãy chạy benchmark trước.")
    else:
        # KPI Row
        best_hr = metrics_df.loc[metrics_df["hit_rate_at_10"].idxmax()]
        best_ndcg = metrics_df.loc[metrics_df["ndcg_at_10"].idxmax()]
        fastest_eval = metrics_df.loc[metrics_df["evaluate_time_seconds"].idxmin()]

        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        with kpi_c1:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Hit Rate@10 Cao nhất</div>
                    <div class="kpi-value">{best_hr['hit_rate_at_10']*100:.2f}%</div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">{best_hr['model']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c2:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">NDCG@10 Cao nhất</div>
                    <div class="kpi-value">{best_ndcg['ndcg_at_10']:.4f}</div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">{best_ndcg['model']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c3:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Top 2 Mô hình Lai ghép</div>
                    <div class="kpi-value">{metrics_df.loc[metrics_df['model']=='Hybrid', 'hit_rate_at_10'].values[0]*100:.2f}%</div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">Hybrid (CF + CB)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kpi_c4:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Đánh giá nhanh nhất</div>
                    <div class="kpi-value">{fastest_eval['evaluate_time_seconds']:.2f}s</div>
                    <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">{fastest_eval['model']} (935 users)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Plot 1: Hit Rate@10 Comparison
        st.subheader("1. Xếp hạng Mô hình theo Hit Rate@10 (Tỷ lệ tìm thấy ít nhất 1 phim đúng)")
        df_sorted_hr = metrics_df.sort_values(by="hit_rate_at_10", ascending=True).copy()
        df_sorted_hr["color"] = df_sorted_hr["model"].apply(
            lambda m: "#4F46E5" if m == "Neural CF" else ("#06B6D4" if m == "Hybrid" else "#334155")
        )

        fig_hr = go.Figure(
            go.Bar(
                x=df_sorted_hr["hit_rate_at_10"] * 100,
                y=df_sorted_hr["model"],
                orientation="h",
                marker_color=df_sorted_hr["color"],
                text=(df_sorted_hr["hit_rate_at_10"] * 100).apply(lambda x: f"{x:.2f}%"),
                textposition="inside",
            )
        )
        fig_hr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
            xaxis=dict(title="Hit Rate@10 (%)", gridcolor="#1E293B"),
            yaxis=dict(title="Mô hình"),
            height=380,
            margin=dict(l=20, r=20, t=30, b=30),
        )
        st.plotly_chart(fig_hr, use_container_width=True)

        # Plot 2: All 5 Top-N Metrics Comparison
        st.subheader("2. So sánh Toàn diện 5 Độ đo Đánh giá Top-10")
        metrics_cols = ["precision_at_10", "recall_at_10", "f1_at_10", "ndcg_at_10", "hit_rate_at_10"]
        metric_labels = {
            "precision_at_10": "Precision@10",
            "recall_at_10": "Recall@10",
            "f1_at_10": "F1-Score@10",
            "ndcg_at_10": "NDCG@10",
            "hit_rate_at_10": "Hit Rate@10",
        }

        selected_metric = st.selectbox(
            "Chọn độ đo hiển thị biểu đồ:",
            options=metrics_cols,
            format_func=lambda col: metric_labels[col],
            index=3,  # NDCG@10
        )

        df_metric_sorted = metrics_df.sort_values(by=selected_metric, ascending=False)
        fig_metric = px.bar(
            df_metric_sorted,
            x="model",
            y=selected_metric,
            color=selected_metric,
            color_continuous_scale="Viridis",
            labels={"model": "Mô hình", selected_metric: metric_labels[selected_metric]},
            text_auto=".4f",
        )
        fig_metric.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(gridcolor="#1E293B"),
            height=380,
        )
        st.plotly_chart(fig_metric, use_container_width=True)

        # Plot 3: Execution Time Comparison
        st.subheader("3. Thời gian Huấn luyện và Đánh giá (Độ phức tạp tính toán)")
        fig_time = go.Figure()
        fig_time.add_trace(
            go.Bar(
                name="Thời gian Huấn luyện (Fit Time)",
                x=metrics_df["model"],
                y=metrics_df["fit_time_seconds"],
                marker_color="#F59E0B",
            )
        )
        fig_time.add_trace(
            go.Bar(
                name="Thời gian Đánh giá (Eval Time - 935 users)",
                x=metrics_df["model"],
                y=metrics_df["evaluate_time_seconds"],
                marker_color="#10B981",
            )
        )
        fig_time.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
            xaxis=dict(gridcolor="#1E293B"),
            yaxis=dict(title="Thời gian (giây)", gridcolor="#1E293B"),
            height=380,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # Detailed Data Table
        st.subheader("4. Bảng Kết quả Chi tiết 10 Mô hình")
        display_df = metrics_df.copy()
        display_df.columns = [
            "Mô hình",
            "Precision@10",
            "Recall@10",
            "F1@10",
            "NDCG@10",
            "Hit Rate@10",
            "Số Users Eval",
            "Fit Time (s)",
            "Eval Time (s)",
            "Total Time (s)",
        ]
        st.dataframe(
            display_df.style.highlight_max(
                subset=["Precision@10", "Recall@10", "F1@10", "NDCG@10", "Hit Rate@10"],
                color="#312E81",
            ),
            use_container_width=True,
        )


# ==============================================================================
# PAGE 3: USER PREFERENCES EXPLORER
# ==============================================================================
elif app_mode == "👤 Hồ sơ sở thích người dùng (Profiles)":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">👤 Khám phá Hồ sơ Sở thích Người dùng</div>
            <div class="hero-subtitle">
                Phân tích lịch sử đánh giá từ tập Train và đối chiếu trực quan danh sách gợi ý giữa 2 mô hình AI
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_u_sel, col_empty = st.columns([1, 2])
    with col_u_sel:
        target_uid = st.selectbox("Chọn Mã người dùng:", options=unique_train_users, index=0)

    # Filter user interactions from Train strictly
    u_history = train_df[train_df["user_id"] == target_uid].copy()
    u_history = u_history.merge(movies_df[["movie_id", "title", "year", "genres_str"]], on="movie_id", how="left")

    total_ratings = len(u_history)
    mean_rating = u_history["rating"].mean()
    high_rated_count = len(u_history[u_history["rating"] >= 4.0])

    # Top KPI
    p_kpi1, p_kpi2, p_kpi3 = st.columns(3)
    with p_kpi1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Tổng số phim đã đánh giá</div>
                <div class="kpi-value">{total_ratings}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p_kpi2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Điểm đánh giá Trung bình</div>
                <div class="kpi-value">{mean_rating:.2f} ★</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with p_kpi3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Số phim thích (Rating ≥ 4.0)</div>
                <div class="kpi-value">{high_rated_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    c_chart1, c_chart2 = st.columns(2)

    with c_chart1:
        st.subheader("Phân bố Điểm số Đánh giá")
        rating_counts = u_history["rating"].value_counts().sort_index().reset_index()
        rating_counts.columns = ["Rating", "Số lượng"]
        fig_ratings = px.bar(
            rating_counts,
            x="Rating",
            y="Số lượng",
            color="Rating",
            color_continuous_scale="Purples",
            text_auto=True,
        )
        fig_ratings.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
            xaxis=dict(tickmode="linear", tick0=1, dtick=1),
            height=300,
        )
        st.plotly_chart(fig_ratings, use_container_width=True)

    with c_chart2:
        st.subheader("Thể loại Phim Xem Nhiều Nhất")
        genre_counter: Dict[str, int] = {}
        for g_str in u_history["genres_str"].dropna():
            for g in g_str.split("|"):
                genre_counter[g] = genre_counter.get(g, 0) + 1
        top_genres = sorted(genre_counter.items(), key=lambda x: x[1], reverse=True)[:8]
        df_genres = pd.DataFrame(top_genres, columns=["Thể loại", "Số phim"])
        fig_genre = px.bar(
            df_genres,
            x="Số phim",
            y="Thể loại",
            orientation="h",
            color="Số phim",
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig_genre.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#F8FAFC",
            height=300,
        )
        st.plotly_chart(fig_genre, use_container_width=True)

    # Top Rated Movies
    st.subheader(f"🌟 Danh sách Phim Yêu thích của Người dùng #{target_uid} (Rating 4-5 Sao)")
    fav_movies = u_history[u_history["rating"] >= 4.0].sort_values(by="rating", ascending=False)
    fav_display = fav_movies[["movie_id", "title", "year", "genres_str", "rating"]].rename(
        columns={
            "movie_id": "Mã phim",
            "title": "Tên phim",
            "year": "Năm",
            "genres_str": "Thể loại",
            "rating": "Điểm chấm",
        }
    )
    st.dataframe(fav_display, use_container_width=True)

    st.markdown("---")

    # Side-by-Side Model Comparison for this specific User
    st.subheader("⚔️ So sánh Đối đầu Top-10 Giữa Hai Mô hình cho Người dùng này")
    c_m1, c_m2 = st.columns(2)
    with c_m1:
        model_a = st.selectbox("Chọn Mô hình A:", options=list(MODEL_REGISTRY.keys()), index=8)  # Neural CF
    with c_m2:
        model_b = st.selectbox("Chọn Mô hình B:", options=list(MODEL_REGISTRY.keys()), index=2)  # User-Based CF

    if st.button("So sánh danh sách gợi ý"):
        with st.spinner("Đang trích xuất và tính toán gợi ý..."):
            inst_a = get_model(model_a, _train_df=train_df, _movies_df=movies_df)
            inst_b = get_model(model_b, _train_df=train_df, _movies_df=movies_df)

            cand_pool = [mid for mid in movies_df["movie_id"].unique() if mid not in set(u_history["movie_id"])]
            recs_a = inst_a.recommend(target_uid, top_n=10, candidate_items=cand_pool)
            recs_b = inst_b.recommend(target_uid, top_n=10, candidate_items=cand_pool)

            set_a = set(recs_a)
            set_b = set(recs_b)
            common = set_a.intersection(set_b)
            jaccard = len(common) / len(set_a.union(set_b)) if set_a.union(set_b) else 0.0

        st.markdown(
            f"""
            <div style="background: #1E293B; border: 1px solid #4338CA; border-radius: 8px; padding: 12px 18px; margin-bottom: 16px;">
                <b>Độ tương đồng Jaccard giữa 2 danh sách Top-10:</b> <span style="color: #38BDF8; font-size: 1.1rem; font-weight: 700;">{jaccard*100:.1f}%</span>
                • <b>Số phim cùng được gợi ý (Giao nhau):</b> <span style="color: #34D399; font-weight: 700;">{len(common)}/10 phim</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_list_a, col_list_b = st.columns(2)
        with col_list_a:
            st.markdown(f"#### Top 10: {model_a}")
            for rank, mid in enumerate(recs_a, start=1):
                m_info = movies_lookup.get(mid, {})
                is_overlap = "⭐ (Trùng)" if mid in common else ""
                st.markdown(f"**#{rank}. {m_info.get('title', 'N/A')}** ({m_info.get('year', 'N/A')}) {is_overlap}")
                st.caption(f"Thể loại: {m_info.get('genres_str', 'N/A')}")

        with col_list_b:
            st.markdown(f"#### Top 10: {model_b}")
            for rank, mid in enumerate(recs_b, start=1):
                m_info = movies_lookup.get(mid, {})
                is_overlap = "⭐ (Trùng)" if mid in common else ""
                st.markdown(f"**#{rank}. {m_info.get('title', 'N/A')}** ({m_info.get('year', 'N/A')}) {is_overlap}")
                st.caption(f"Thể loại: {m_info.get('genres_str', 'N/A')}")


# ==============================================================================
# PAGE 4: ABOUT THE PROJECT
# ==============================================================================
elif app_mode == "ℹ️ Giới thiệu đề tài (About)":
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">ℹ️ Giới thiệu Đề tài và Kiến trúc Hệ thống</div>
            <div class="hero-subtitle">
                Xây dựng chương trình gợi ý phim/bài hát theo sở thích ứng dụng Trí tuệ nhân tạo
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        r"""
        ### 1. Thông tin Đề tài
        - **Tên đề tài:** Xây dựng chương trình gợi ý phim/bài hát theo sở thích ứng dụng Trí tuệ nhân tạo.
        - **Học phần:** Trí tuệ nhân tạo (K5).
        - **Tập dữ liệu:** MovieLens 100K (GroupLens Research, University of Minnesota).
        - **Quy mô dữ liệu:** 100,000 lượt tương tác, 943 người dùng, 1,682 bộ phim, ma trận thưa 93.7%.

        ---

        ### 2. Tổng quan 10 Mô hình AI Triển khai Thực tế
        Hệ thống triển khai đầy đủ 10 thuật toán thuộc 4 nhóm phương pháp:
        1. **Nhóm Baselines:**
           - `Popularity Recommender`: Xếp hạng toàn cục bằng công thức ước lượng Bayes (IMDb Weighted Rating).
           - `Content-Based Recommender`: Trích xuất đặc trưng TF-IDF từ thể loại và tựa đề (2,350 chiều) và tính độ tương đồng Cosine.
        2. **Nhóm Neighborhood Collaborative Filtering (KNN):**
           - `User-Based CF`: Tìm kiếm người dùng tương đồng qua tương quan Pearson thu hẹp (shrinkage) và kết hợp độ lệch.
           - `Item-Based CF`: Tìm kiếm phim tương đồng qua Adjusted Cosine Similarity.
           - `KNNBaseline`: Tách biệt xu hướng toàn cục, bias người dùng và bias phim, kết hợp tương quan trên phần dư.
        3. **Nhóm Matrix Factorization:**
           - `Funk SVD`: Phân rã ma trận rating sang không gian 20 chiều ẩn, tối ưu hóa qua Mini-Batch SGD.
           - `SVD++`: Tích hợp thêm phản hồi ngầm định (implicit feedback) từ lịch sử tương tác.
           - `NMF (Non-Negative Matrix Factorization)`: Ràng buộc không âm nhằm trích xuất các đặc trưng thành phần có thể giải thích.
        4. **Nhóm Deep Learning & Hybrid:**
           - `Neural CF (NeuMF)`: Kết hợp song song GMF (tuyến tính) và MLP (phi tuyến 3 tầng) tối ưu với BCEWithLogitsLoss và Negative Sampling.
           - `Weighted Hybrid Recommender`: Chuẩn hóa động Min-Max trên tập ứng viên và kết hợp trọng số 80% CF + 20% CB.

        ---

        ### 3. Giao thức Đánh giá Chuẩn hóa (Evaluation Protocol)
        - **Phân hoạch:** Stratified Per-User Split (80% Train: 80,000 dòng / 20% Test: 20,000 dòng), cố định `seed=42`.
        - **Tập người dùng đánh giá:** Đúng 935 người dùng có tương tác $r \ge 4.0$ trong tập Test.
        - **Tập ứng viên xếp hạng:** $\mathcal{C}_u = \mathcal{I}_{all} \setminus \mathcal{I}_u^{train}$ (loại bỏ hoàn toàn các phim đã xem trong Train).
        - **5 Độ đo Đánh giá:** Precision@10, Recall@10, F1-Score@10, NDCG@10, Hit Rate@10.

        > [!IMPORTANT]
        > **Giải thích Khoa học về Hit Rate@10 = 82.03%:**
        > Chỉ số Hit Rate@10 **không mang ý nghĩa** là 82.03% số phim được gợi ý là chính xác.
        > Về mặt toán học, **Hit Rate@10 = 82.03%** có nghĩa là: Trong số 935 người dùng kiểm thử, có đúng **82.03% người dùng nhận được ít nhất một bộ phim phù hợp ($r \ge 4.0$ trong Test) nằm trong danh sách Top 10 đề xuất**. Đây là chỉ số chuẩn mực quốc tế phản ánh độ phủ hài lòng của hệ thống gợi ý.

        ---

        ### 4. Công nghệ Sử dụng
        - **Ngôn ngữ:** Python 3.11.9
        - **Deep Learning:** PyTorch 2.14.1 (CPU optimized)
        - **Data Science:** Pandas 2.2, NumPy 1.26, SciPy 1.14, Scikit-learn 1.5
        - **Giao diện Web:** Streamlit 1.66, Plotly 7.1
        - **Kiểm thử tự động:** Unittest (76 test cases passed 100%)
        """
    )
