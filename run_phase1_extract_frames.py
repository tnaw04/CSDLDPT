"""
╔══════════════════════════════════════════════════════════════╗
║  PHASE 1 — TRÍCH XUẤT KEYFRAME (Frame Extraction)           ║
║                                                              ║
║  Nhiệm vụ: Đọc Video → Phân cảnh → Lưu ảnh .jpg ra disk    ║
║  Thư viện: cv2, numpy, os (KHÔNG dùng CLIP, psycopg2)       ║
║                                                              ║
║  Cách chạy:                                                  ║
║    python run_phase1_extract_frames.py                       ║
║    python run_phase1_extract_frames.py --start 0 --end 50   ║
║    python run_phase1_extract_frames.py --force               ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse
import glob
import time

# ── Thiết lập đường dẫn ─────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.abspath(__file__))

RAW_VIDEO_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "animal_videos")
FRAMES_DIR    = os.path.join(PROJECT_ROOT, "data", "frames")   # thư mục ảnh Phase 1

os.makedirs(FRAMES_DIR, exist_ok=True)


# ── Tiện ích ─────────────────────────────────────────────────────────────────
def get_video_files():
    exts  = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    files = []
    for ext in exts:
        files += glob.glob(os.path.join(RAW_VIDEO_DIR, ext))
    return sorted(files)


def already_processed(video_name: str) -> int:
    """Trả về số keyframes hiện có của video (0 = chưa xử lý)."""
    existing = glob.glob(os.path.join(FRAMES_DIR, f"{video_name}_frame*.jpg"))
    return len(existing)


# ── Hàm chính Phase 1 ────────────────────────────────────────────────────────
def run_phase1(video_files: list, force: bool = False):
    from src.keyframe_extractor import KeyframeExtractor

    print("\n" + "=" * 60)
    print("  PHASE 1: TRÍCH XUẤT KEYFRAMES TỪ VIDEO")
    print("=" * 60)
    print(f"  📂 Nguồn video  : {RAW_VIDEO_DIR}")
    print(f"  💾 Lưu frames   : {FRAMES_DIR}")
    print(f"  🎞  Tổng số video: {len(video_files)}")
    print("=" * 60 + "\n")

    extractor = KeyframeExtractor()   # chỉ dùng TransNetV2 + cv2 + histogram
    total = len(video_files)
    success_count = 0
    skip_count    = 0
    error_count   = 0

    for i, vpath in enumerate(video_files, 1):
        vname = os.path.splitext(os.path.basename(vpath))[0]

        # ── Cơ chế "Lưu điểm nhớ" (Checkpointing) ─────────────────────────
        n_existing = already_processed(vname)
        if n_existing > 0 and not force:
            print(f"[{i:3d}/{total}] ⏭  Bỏ qua '{vname}' (đã có {n_existing} frames)")
            skip_count += 1
            continue

        print(f"[{i:3d}/{total}] 🎬 Đang xử lý: {vname}")
        t0 = time.time()
        try:
            saved = extractor.extract_keyframes(vpath, FRAMES_DIR)
            elapsed = time.time() - t0
            print(f"  ✅ Xong! Lưu được {saved} keyframes  ⏱ {elapsed:.1f}s")
            success_count += 1
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ❌ Lỗi khi xử lý '{vname}': {e}  ⏱ {elapsed:.1f}s")
            error_count += 1

    # ── Tổng kết ─────────────────────────────────────────────────────────────
    total_frames = len(glob.glob(os.path.join(FRAMES_DIR, "*.jpg")))
    print("\n" + "=" * 60)
    print("  KẾT QUẢ PHASE 1")
    print("=" * 60)
    print(f"  ✅ Thành công : {success_count} video")
    print(f"  ⏭  Đã bỏ qua : {skip_count} video (đã xử lý trước đó)")
    print(f"  ❌ Lỗi       : {error_count} video")
    print(f"  🖼  Tổng frames: {total_frames} ảnh .jpg trong {FRAMES_DIR}")
    print("=" * 60)
    print("\n👉 BƯỚC TIẾP THEO: Kiểm tra ảnh trong thư mục data/frames/")
    print("   Nếu ảnh ổn, hãy chạy Phase 2:")
    print("   python run_phase2_build_database.py")


# ── Entry Point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phase 1 — Trích xuất keyframe từ video, lưu ra ảnh .jpg"
    )
    parser.add_argument("--start", type=int, default=0,    help="Index video bắt đầu (0-indexed)")
    parser.add_argument("--end",   type=int, default=None, help="Index video kết thúc (exclusive)")
    parser.add_argument("--force", action="store_true",    help="Xử lý lại tất cả dù đã có keyframe")
    args = parser.parse_args()

    video_files = get_video_files()
    if not video_files:
        print(f"⚠️  Không tìm thấy video nào trong: {RAW_VIDEO_DIR}")
        sys.exit(1)

    subset = video_files[args.start : args.end]
    print(f"🚀 Khởi động Phase 1 — xử lý {len(subset)} video")
    if args.force:
        print("⚠️  Chế độ --force: sẽ xử lý lại tất cả video!")

    run_phase1(subset, force=args.force)


if __name__ == "__main__":
    main()
