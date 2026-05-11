-- ══════════════════════════════════════════════════════════════
-- Schema cho Video Retrieval System (PostgreSQL + pgvector)
-- Chạy lệnh này một lần trước khi dùng Phase 2:
--   psql -U postgres -d video_retrieval -f db/schema.sql
-- ══════════════════════════════════════════════════════════════

-- 1. Cài extension pgvector (chạy 1 lần với tài khoản superuser)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Bảng lưu trữ keyframes và vector đặc trưng
CREATE TABLE IF NOT EXISTS keyframes (
    id           SERIAL PRIMARY KEY,
    video_name   TEXT        NOT NULL,
    frame_path   TEXT        UNIQUE NOT NULL,   -- đường dẫn tuyệt đối tới ảnh .jpg
    frame_pos    INTEGER     NOT NULL,           -- vị trí frame trong video
    clip_vector  vector(512) NOT NULL,           -- CLIP ViT-B/32 semantic vector
    color_vector vector(512) NOT NULL,           -- HSV histogram color vector
    created_at   TIMESTAMP   DEFAULT NOW()
);

-- 3. Index tìm kiếm cosine trên CLIP vector (dùng IVFFlat cho tốc độ)
--    lists = 100 phù hợp cho ~10k–100k vectors
CREATE INDEX IF NOT EXISTS idx_clip_vector
ON keyframes USING ivfflat (clip_vector vector_cosine_ops)
WITH (lists = 100);

-- 4. Index tìm kiếm cosine trên Color vector
CREATE INDEX IF NOT EXISTS idx_color_vector
ON keyframes USING ivfflat (color_vector vector_cosine_ops)
WITH (lists = 100);

-- 5. Index thường để lọc theo video_name nhanh
CREATE INDEX IF NOT EXISTS idx_video_name
ON keyframes (video_name);

-- ── Kiểm tra sau khi chạy ────────────────────────────────────
-- SELECT COUNT(*) FROM keyframes;
-- SELECT * FROM keyframes LIMIT 5;
