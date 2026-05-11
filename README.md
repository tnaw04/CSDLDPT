# 🎬 Video Retrieval System — Decoupled Pipeline

Hệ thống tìm kiếm video bằng ảnh/văn bản, sử dụng kiến trúc **Decoupled Pipeline** (Luồng xử lý tách rời) để tối ưu RAM 8GB.

---

## 📁 Cấu trúc thư mục

```
video_retrieval_project/
│
├── run_phase1_extract_frames.py   ← CHẠY TRƯỚC: Cắt video → lưu ảnh .jpg
├── run_phase2_build_database.py   ← CHẠY SAU : Trích vector → nạp PostgreSQL
│
├── src/
│   ├── keyframe_extractor.py      # Phase 1: TransNetV2 / Histogram (chỉ cv2+numpy)
│   ├── feature_extractor.py       # Phase 2: CLIP + Color Histogram
│   ├── db_writer.py               # Phase 2: Ghi vào PostgreSQL
│   ├── index_builder.py           # Phase 2: Tìm kiếm Late Fusion với pgvector
│   └── search_engine.py           # (sắp triển khai) API tìm kiếm online
│
├── config/
│   └── db_config.py               # ← SỬA: điền thông tin PostgreSQL của bạn
│
├── db/
│   └── schema.sql                 # Tạo bảng & index pgvector (chạy 1 lần)
│
└── data/
    ├── raw/
    │   └── animal_videos/         # Đặt video .mp4 vào đây
    ├── frames/                    # Phase 1 → lưu ảnh .jpg ra đây
    └── queries/                   # Ảnh query để tìm kiếm
```

---

## 🚀 Hướng dẫn chạy

### Bước 0: Chuẩn bị

```bash
# Cài thư viện
pip install -r requirements.txt

# Cài pgvector cho PostgreSQL (chạy trong psql):
# CREATE EXTENSION vector;

# Tạo schema database
psql -U postgres -d video_retrieval -f db/schema.sql

# Sửa thông tin kết nối
# → config/db_config.py
```

### Bước 1: Trích xuất Keyframes (Phase 1)

```bash
# Chạy toàn bộ thư mục video
python run_phase1_extract_frames.py

# Chỉ chạy 50 video đầu tiên
python run_phase1_extract_frames.py --start 0 --end 50

# Chạy lại từ đầu (bỏ qua checkpoint)
python run_phase1_extract_frames.py --force
```

Sau khi chạy: **Kiểm tra ảnh** trong `data/frames/` bằng File Explorer.  
Nếu thấy ảnh đen/mờ/sai → sửa code Phase 1 → chạy lại với `--force`.

### Bước 2: Xây dựng Database (Phase 2)

```bash
# Nạp toàn bộ ảnh vào PostgreSQL
python run_phase2_build_database.py

# Tiếp tục nếu bị ngắt giữa chừng (Resume mode)
python run_phase2_build_database.py --resume

# Điều chỉnh batch size cho máy yếu RAM
python run_phase2_build_database.py --batch-size 16
```

---

## 💡 Tại sao dùng Decoupled Pipeline?

| Vấn đề | Gộp chung 1 file | Tách Phase |
|--------|-----------------|------------|
| RAM 8GB | Tranh giành → tràn | Từng Phase dùng riêng |
| Lỗi giữa chừng | Mất hết, chạy lại từ đầu | Resume từ điểm dừng |
| Kiểm tra ảnh | Không thể | Mở thư mục xem ngay |
| Ổ cứng | Không tốn | ~1–5GB ảnh tạm thời |

---

## ⚙️ Yêu cầu hệ thống

- Python 3.10+
- PostgreSQL 14+ với extension **pgvector**
- RAM: 8GB (tối thiểu)
- GPU: Tuỳ chọn (CLIP chạy được trên CPU, chậm hơn ~5x)
