"""
setup_env.py — Thiết lập môi trường cho Video Retrieval Project
================================================================
Chạy script này một lần để tạo cấu trúc thư mục và hướng dẫn cài đặt.
"""

import os
import subprocess

def setup():
    print("⏳ Bắt đầu thiết lập môi trường cho Video Retrieval Project...")

    # 1. Tạo cấu trúc thư mục
    dirs = [
        r"data\raw\animal_videos",
        r"data\frames",          # Phase 1 output
        r"data\processed",       # ảnh đã xử lý (tương thích cũ)
        r"data\queries",         # ảnh query để tìm kiếm
        r"config",
        r"db",
        r"models",
        r"notebooks",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ Đã tạo các thư mục cần thiết")

    # 2. Nhắc cài thư viện
    print("\n📦 Cài đặt thư viện:")
    print("   pip install -r requirements.txt")
    print("\n📦 Cài thêm nếu muốn dùng TransNetV2 (tùy chọn):")
    print("   pip install transnetv2-pytorch")

    # 3. Nhắc cài pgvector
    print("\n🐘 Cài pgvector cho PostgreSQL (chạy trong psql):")
    print("   CREATE EXTENSION vector;")
    print("\n🐘 Tạo schema database:")
    print("   psql -U postgres -d video_retrieval -f db/schema.sql")

    # 4. Nhắc cấu hình DB
    print("\n⚙️  Điền thông tin kết nối PostgreSQL:")
    print("   Mở file config/db_config.py và sửa password")

    print("\n✅ Môi trường đã sẵn sàng!")
    print("\n🚀 Quy trình chạy:")
    print("   1. python run_phase1_extract_frames.py   ← Cắt video → ảnh .jpg")
    print("   2. (Kiểm tra ảnh trong data/frames/)")
    print("   3. python run_phase2_build_database.py   ← Vector → PostgreSQL")


if __name__ == "__main__":
    project_root = r"c:\Crawl\video_retrieval_project"
    if os.getcwd().lower() != project_root.lower():
        os.chdir(project_root)
    setup()
