"""
PHASE 2 — IndexBuilder (PostgreSQL + pgvector)
===============================================
Thay thế hoàn toàn FAISS.

Tìm kiếm dùng pgvector cosine operator (<=>):
  clip_vector <=> query_vec    (cosine distance)
"""

import numpy as np
import psycopg2
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "config"))

try:
    from db_config import DB_CONFIG
except ImportError:
    DB_CONFIG = {
        "host": "localhost", "port": 5432,
        "database": "video_retrieval",
        "user": "postgres", "password": "postgres",
    }


class IndexBuilder:
    """
    Lớp truy vấn vector từ PostgreSQL (thay thế FAISS IndexFlatIP).

    Tìm kiếm Late Fusion:
        score = w_clip * (1 - clip_dist) + w_color * (1 - color_dist)
    """

    def __init__(self, w_clip: float = 0.7, w_color: float = 0.3):
        """
        Parameters:
            w_clip  : Trọng số cho semantic CLIP score  (mặc định 0.7)
            w_color : Trọng số cho color histogram score (mặc định 0.3)
        """
        self.w_clip  = w_clip
        self.w_color = w_color
        self.conn    = None

    def connect(self):
        self.conn = psycopg2.connect(**DB_CONFIG)

    def close(self):
        if self.conn:
            self.conn.close()

    # ──────────────────────────────────────────────────────────────────────────
    def search(
        self,
        clip_query:  np.ndarray,
        color_query: np.ndarray,
        top_k:       int = 20,
    ) -> list[dict]:
        """
        Tìm kiếm Late Fusion: kết hợp CLIP + Color distance từ PostgreSQL.

        Returns:
            list of dict: {id, video_name, frame_path, frame_pos, score}
            Sắp xếp giảm dần theo score (1.0 = hoàn toàn giống).
        """
        clip_list  = clip_query.tolist()
        color_list = color_query.tolist()

        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    video_name,
                    frame_path,
                    frame_pos,
                    (clip_vector  <=> %s::vector) AS clip_dist,
                    (color_vector <=> %s::vector) AS color_dist
                FROM keyframes
                ORDER BY
                    %s * (clip_vector <=> %s::vector)
                  + %s * (color_vector <=> %s::vector)
                ASC
                LIMIT %s;
                """,
                (
                    clip_list, color_list,
                    self.w_clip,  clip_list,
                    self.w_color, color_list,
                    top_k,
                ),
            )
            rows = cur.fetchall()

        results = []
        for row in rows:
            rid, vname, fpath, fpos, clip_dist, color_dist = row
            score = self.w_clip * (1 - clip_dist) + self.w_color * (1 - color_dist)
            results.append({
                "id":         rid,
                "video_name": vname,
                "frame_path": fpath,
                "frame_pos":  fpos,
                "score":      float(score),
            })

        return results

    # ──────────────────────────────────────────────────────────────────────────
    def count(self) -> int:
        """Trả về số keyframes đang có trong database."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM keyframes;")
            return cur.fetchone()[0]
