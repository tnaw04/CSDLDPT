import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
import cv2
from PIL import Image
from rembg import remove

# ── HÀM TIỀN XỬ LÝ ẢNH TRUY VẤN (REMBG U^2-NET) ──────────────────────────
def ultimate_segment_animal(frame, target_size=(512, 512)):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        output_rgba = remove(img_rgb)
    except Exception:
        output_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    
    alpha_channel = output_rgba[:, :, 3]
    mask = (alpha_channel > 50).astype(np.uint8)
    foreground = cv2.cvtColor(output_rgba, cv2.COLOR_RGBA2BGR)
    segmented_img = cv2.bitwise_and(foreground, foreground, mask=mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h_box = cv2.boundingRect(c)
        
        h, w_img = frame.shape[:2]
        mx, my = int(w * 0.05), int(h_box * 0.05)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(w_img, x + w + mx), min(h, y + h_box + my)
        cropped_img = segmented_img[y1:y2, x1:x2]
    else:
        cropped_img = segmented_img

    ch, cw = cropped_img.shape[:2]
    if ch == 0 or cw == 0:
        return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        
    scale = min(target_size[1] / cw, target_size[0] / ch)
    new_w, new_h = int(cw * scale), int(ch * scale)
    
    resized_crop = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    x_offset = (target_size[1] - new_w) // 2
    y_offset = (target_size[0] - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_crop
    
    return canvas

# ── CẤU HÌNH TRANG WEB STREAMLIT ──────────────────────────────────────────────
st.set_page_config(
    page_title="Video Retrieval System (CBVR)",
    page_icon="🔍",
    layout="wide"
)

# ── ĐƯỜNG DẪN DỮ LIỆU (PHASE 2) ─────────────────────────────────────────────
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "index")
MAPPING_CSV = os.path.join(INDEX_DIR, "mapping.csv")
FAISS_CLIP_IDX = os.path.join(INDEX_DIR, "faiss_clip.index")
FAISS_COLOR_IDX = os.path.join(INDEX_DIR, "faiss_color.index")
FAISS_HOG_IDX = os.path.join(INDEX_DIR, "faiss_hog.index")
FAISS_TEX_IDX = os.path.join(INDEX_DIR, "faiss_texture.index")

# ── KHỐI QUẢN LÝ BỘ NHỚ (CỐT LÕI CHO MÁY 8GB RAM) ───────────────────────────

@st.cache_resource
def load_feature_extractor():
    """Load bộ trích xuất 4 đặc trưng (CLIP, Color, HOG, LBP)."""
    from src.feature_extractor import FeatureExtractor
    return FeatureExtractor()

@st.cache_resource
def load_faiss_indices():
    """Tải 4 kho FAISS lên RAM."""
    indices = {}
    paths = {
        'clip': FAISS_CLIP_IDX,
        'color': FAISS_COLOR_IDX,
        'hog': FAISS_HOG_IDX,
        'tex': FAISS_TEX_IDX
    }
    for name, path in paths.items():
        if os.path.exists(path):
            indices[name] = faiss.read_index(path)
        else:
            indices[name] = None
    return indices

@st.cache_resource
def load_metadata():
    """Đọc file mapping."""
    if not os.path.exists(MAPPING_CSV):
        return None
    return pd.read_csv(MAPPING_CSV)

with st.spinner("⏳ Đang tải mô hình AI và CSDL Vector vào RAM (Chỉ tải 1 lần đầu)..."):
    extractor = load_feature_extractor()
    indices = load_faiss_indices()
    metadata_df = load_metadata()

# ── GIAO DIỆN CHÍNH (MAIN UI) ────────────────────────────────────────────────

st.title("🔍 HỆ THỐNG TÌM KIẾM VIDEO (LATE FUSION)")
st.write("Giải pháp tối ưu RAM - Tích hợp Tứ trụ Đặc trưng (CLIP, Color, HOG, Texture LBP) xếp hạng Video.")

if None in indices.values() or metadata_df is None:
    st.error(f"⚠️ Không tìm thấy Database. Hãy chạy Phase 1 & Phase 2 trước.")
    st.stop()

# ── THANH BÊN (SIDEBAR) ĐIỀU CHỈNH TRỌNG SỐ ─────────────────────────────────
st.sidebar.header("🎛️ Bảng Điều Khiển Late Fusion")
st.sidebar.write("Chỉnh trọng số $\\alpha, \\beta, \\gamma, \\delta$ cho từng đặc trưng:")

w_clip = st.sidebar.slider("🧠 Ngữ Nghĩa (CLIP)", 0.0, 2.0, 1.0, step=0.1)
w_color = st.sidebar.slider("🎨 Màu Sắc (Color HSV)", 0.0, 2.0, 0.5, step=0.1)
w_hog = st.sidebar.slider("📐 Hình Dáng (HOG)", 0.0, 2.0, 0.5, step=0.1)
w_tex = st.sidebar.slider("🦓 Kết Cấu Lông/Vằn (LBP Texture)", 0.0, 2.0, 1.0, step=0.1)

# Khu vực Upload Ảnh
st.markdown("### 📥 1. Upload ảnh truy vấn")
uploaded_file = st.file_uploader("Kéo thả hoặc chọn ảnh (Hỗ trợ JPG, PNG)...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col_query, col_results = st.columns([1, 2.5])
    
    # Ghi tạm ảnh ra ổ cứng để FeatureExtractor (cv2.imread) có thể đọc
    temp_path = "temp_query.jpg"
    query_image = Image.open(uploaded_file).convert("RGB")
    query_image.save(temp_path)
    
    with col_query:
        st.markdown("**Ảnh bạn vừa tải lên:**")
        st.image(query_image, use_container_width=True)
        
        if st.button("🚀 BẮT ĐẦU TÌM KIẾM VIDEO", type="primary", use_container_width=True):
            with st.spinner("Đang tách nền bẳng U^2-Net, tính toán Vector & Quét FAISS..."):
                
                # --- BƯỚC 0: TIỀN XỬ LÝ QUERY BẰNG REMBG ĐỂ ĐỒNG BỘ DATASET ---
                cv_img = cv2.imread(temp_path)
                processed_cv = ultimate_segment_animal(cv_img)
                cv2.imwrite(temp_path, processed_cv) # Ghi đè temp_path bằng ảnh đã xóa nền
                
                # --- BƯỚC 1: TRÍCH XUẤT 4 VECTOR ---
                clip_vec, color_vec, hog_vec, tex_vec = extractor.extract(temp_path)
                
                # Hàm query phụ trợ
                def query_all(index_name, vec):
                    vec_2d = np.expand_dims(vec, axis=0)
                    total_db_size = indices[index_name].ntotal
                    D, I = indices[index_name].search(vec_2d, total_db_size)
                    
                    # Dàn phẳng điểm theo ID chuẩn
                    scores = np.zeros(total_db_size)
                    for i in range(total_db_size):
                        frame_id = I[0][i]
                        scores[frame_id] = D[0][i]
                    return scores

                # --- BƯỚC 2: QUÉT ĐIỂM TOÀN BỘ ẢNH TRONG DB ---
                scores_clip = query_all('clip', clip_vec)
                scores_color = query_all('color', color_vec)
                scores_hog = query_all('hog', hog_vec)
                scores_tex = query_all('tex', tex_vec)
                
                # --- BƯỚC 3: LATE FUSION TOÁN HỌC ---
                total_scores = (w_clip * scores_clip) + \
                               (w_color * scores_color) + \
                               (w_hog * scores_hog) + \
                               (w_tex * scores_tex)
                
                # Cập nhật kết quả vào DataFrame
                df_results = metadata_df.copy()
                df_results['total_score'] = total_scores
                df_results['s_clip'] = scores_clip
                df_results['s_color'] = scores_color
                df_results['s_hog'] = scores_hog
                df_results['s_tex'] = scores_tex
                
                # --- BƯỚC 4: NHÓM THEO VIDEO CÓ ĐIỂM FRAME CAO NHẤT ---
                # Lấy index của frame có điểm cao nhất cho mỗi Video
                idx_max = df_results.groupby('video_name')['total_score'].idxmax()
                
                # Lọc ra các Frame đại diện, sắp xếp giảm dần và lấy Top 5
                top_videos_df = df_results.loc[idx_max].sort_values(by='total_score', ascending=False).head(5)
                
                st.session_state['top_videos'] = top_videos_df

    # ── CỘT PHẢI: HIỂN THỊ 5 VIDEO TOP ──────────────────────────────────────
    with col_results:
        if 'top_videos' in st.session_state:
            st.markdown("### 🔥 TOP 5 VIDEO GIỐNG NHẤT (Video Grouping)")
            top_videos_df = st.session_state['top_videos']
            
            res_cols = st.columns(5)
            
            for i in range(len(top_videos_df)):
                row = top_videos_df.iloc[i]
                
                v_name = row['video_name']
                f_path = row['frame_path']
                f_pos  = row['frame_pos']
                
                t_score = row['total_score']
                s_c = row['s_clip']
                s_col = row['s_color']
                s_h = row['s_hog']
                s_t = row['s_tex']
                
                time_sec = f_pos / 30.0
                
                with res_cols[i]:
                    try:
                        st.image(f_path, use_container_width=True)
                        st.markdown(f"**🎥 {v_name}**")
                        st.caption(f"⏱ Phút {time_sec//60:.0f}:{time_sec%60:02.0f}")
                        
                        # Hiển thị điểm thành phần bên dưới
                        st.info(f"**Tổng điểm: {t_score:.2f}**")
                        st.markdown(f"""
                        <div style="font-size: 13px; line-height: 1.4;">
                            🧠 CLIP: <b>{s_c:.2f}</b><br>
                            🎨 Màu: <b>{s_col:.2f}</b><br>
                            📐 HOG: <b>{s_h:.2f}</b><br>
                            🦓 LBP: <b>{s_t:.2f}</b>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error("Lỗi: Không tải được ảnh.")
