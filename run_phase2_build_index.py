"""
╔══════════════════════════════════════════════════════════════╗
║  PHASE 2 — XÂY DỰNG CƠ SỞ DỮ LIỆU (FAISS + CSV)             ║
║                                                              ║
║  Nhiệm vụ: Quét data/frames/ → Trích 3 vector (CLIP, Color, HOG)
║            → Nạp thẳng vào FAISS index                       ║
║            → Ghi metadata vào mapping.csv                    ║
║  Thư viện: clip, torch, cv2, faiss, pandas                   ║
║                                                              ║
║  Cách chạy:                                                  ║
║    python run_phase2_build_index.py                          ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import glob
import time
import csv
import numpy as np
import faiss

# ── Thiết lập đường dẫn ─────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Đổi từ frames sang masks theo kịch bản Phase 1.5 GrabCut
MASKS_DIR   = os.path.join(PROJECT_ROOT, "data", "masks")

INDEX_DIR      = os.path.join(PROJECT_ROOT, "data", "index")
MAPPING_CSV    = os.path.join(INDEX_DIR, "mapping.csv")
FAISS_CLIP_IDX = os.path.join(INDEX_DIR, "faiss_clip.index")
FAISS_COLOR_IDX= os.path.join(INDEX_DIR, "faiss_color.index")
FAISS_HOG_IDX  = os.path.join(INDEX_DIR, "faiss_hog.index")
FAISS_TEX_IDX  = os.path.join(INDEX_DIR, "faiss_texture.index")

os.makedirs(INDEX_DIR, exist_ok=True)

def run_phase2():
    from src.feature_extractor import FeatureExtractor
    
    print("\n" + "=" * 60)
    print("  PHASE 2: XÂY DỰNG DATABASE FAISS TỪ KEYFRAMES")
    print("=" * 60)

    # 1. Lấy danh sách ảnh .jpg từ thư mục masks
    all_frames = sorted(glob.glob(os.path.join(MASKS_DIR, "*.jpg")))
    if not all_frames:
        print(f"⚠️  Không tìm thấy ảnh .jpg nào trong: {MASKS_DIR}")
        print("   → Hãy chạy Phase 1.5 (GrabCut) trước để tạo mask.")
        sys.exit(1)

    print(f"  🖼  Tổng keyframes cần xử lý: {len(all_frames)}\n")

    # 2. Khởi tạo FAISS Indices (Dùng Inner Product vì vectors đã L2-normalized)
    clip_dim = 512
    color_dim = 512
    hog_dim = 3780
    
    index_clip = faiss.IndexFlatIP(clip_dim)
    index_color = faiss.IndexFlatIP(color_dim)
    index_hog = faiss.IndexFlatIP(hog_dim)
    index_tex = faiss.IndexFlatIP(26)
    
    extractor = FeatureExtractor()
    
    # 3. Quét ảnh và lưu CSV
    success_cnt = 0
    error_cnt = 0
    t_start = time.time()
    
    with open(MAPPING_CSV, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vector_id", "video_name", "frame_path", "frame_pos"])
        
        for i, frame_path in enumerate(all_frames):
            frame_name = os.path.basename(frame_path)
            
            # Phân tích tên file: {video_name}_frame{XXXXXX}.jpg
            parts = frame_name.replace(".jpg", "").rsplit("_frame", 1)
            video_name = parts[0]
            frame_pos = int(parts[1]) if len(parts) == 2 else 0
            
            try:
                clip_vec, color_vec, hog_vec, tex_vec = extractor.extract(frame_path)
                
                # Thêm vào index (FAISS yêu cầu mảng 2 chiều batch x dim)
                index_clip.add(np.expand_dims(clip_vec, axis=0))
                index_color.add(np.expand_dims(color_vec, axis=0))
                index_hog.add(np.expand_dims(hog_vec, axis=0))
                index_tex.add(np.expand_dims(tex_vec, axis=0))
                
                # Ghi mapping. FAISS tự động đánh số từ 0 mỗi khi add thành công
                writer.writerow([success_cnt, video_name, frame_path, frame_pos])
                success_cnt += 1
                
            except Exception as e:
                print(f"  ❌ Lỗi trích xuất '{frame_name}': {e}")
                error_cnt += 1
            
            # In tiến trình mỗi 50 frames
            if (i + 1) % 50 == 0 or (i + 1) == len(all_frames):
                elapsed = time.time() - t_start
                pct = (i + 1) / len(all_frames) * 100
                print(f"  [{i+1:5d}/{len(all_frames)}] {pct:5.1f}%  ✅ {success_cnt}  ❌ {error_cnt}  ⏱ {elapsed:.0f}s")
    
    # 4. Ghi FAISS files ra đĩa
    print("\n💾 Đang lưu FAISS index ra ổ cứng...")
    faiss.write_index(index_clip, FAISS_CLIP_IDX)
    faiss.write_index(index_color, FAISS_COLOR_IDX)
    faiss.write_index(index_hog, FAISS_HOG_IDX)
    faiss.write_index(index_tex, FAISS_TEX_IDX)
    
    print("\n" + "=" * 60)
    print("  KẾT QUẢ PHASE 2")
    print("=" * 60)
    print(f"  ✅ Đã trích xuất & lưu FAISS : {success_cnt} keyframes")
    print(f"  ❌ Lỗi                       : {error_cnt} keyframes")
    print(f"  ⏱  Tổng thời gian            : {time.time() - t_start:.1f}s")
    print("=" * 60)
    print("\n✅ PHASE 2 HOÀN TẤT! FAISS Database và mapping.csv sẵn sàng.")

if __name__ == "__main__":
    run_phase2()
