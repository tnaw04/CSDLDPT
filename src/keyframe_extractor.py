"""
PHASE 1 — KeyframeExtractor
============================
Nhiệm vụ: Phân cảnh video và lưu keyframes ra ảnh .jpg.

Thuật toán:
  - Ưu tiên 1: TransNetV2 (nếu cài được) → phân cảnh chính xác
  - Fallback   : Histogram Difference     → nhanh, không cần GPU, không cần thư viện nặng

Thư viện sử dụng: cv2, numpy, os
KHÔNG sử dụng: CLIP, torch (ở đây), psycopg2, faiss
"""

import os
import cv2
import numpy as np
from typing import Optional

try:
    import torch
    from transnetv2_pytorch import TransNetV2 as _TransNetV2
    _HAS_TRANSNET = True
except ImportError:
    _HAS_TRANSNET = False

def preprocess_frame(frame, target_size=(512, 512)):
    """
    Tiền xử lý ảnh: Bóp nhỏ ảnh về chuẩn target_size nhưng KHÔNG làm méo ảnh.
    Phần bị dư sẽ được độn viền đen (Letterboxing / Padding).
    """
    h, w = frame.shape[:2]
    scale = min(target_size[1] / w, target_size[0] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    x_offset = (target_size[1] - new_w) // 2
    y_offset = (target_size[0] - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
    
    return canvas


# ═══════════════════════════════════════════════════════════════════════════════
class KeyframeExtractor:
    """
    Trích xuất Keyframe từ video và lưu ra thư mục đích.

    Thứ tự ưu tiên thuật toán:
      1. TransNetV2 (chính xác nhất — yêu cầu cài transnetv2-pytorch)
      2. Histogram Difference (fallback tự động — chỉ cần opencv)
    """

    def __init__(
        self,
        method: str = "auto",        # "transnet" | "histogram" | "auto"
        hist_threshold: float = 0.4, # Ngưỡng phát hiện scene cut (histogram)
        min_shot_length: int  = 15,  # Bỏ qua shot ngắn hơn X frames (nhiễu)
        max_keyframes_per_shot: int = 3,  # Số keyframe tối đa mỗi shot
        jpeg_quality: int = 90,      # Chất lượng ảnh lưu ra (0-100)
    ):
        self.hist_threshold         = hist_threshold
        self.min_shot_length        = min_shot_length
        self.max_keyframes_per_shot  = max_keyframes_per_shot
        self.jpeg_quality           = jpeg_quality

        # ── Chọn phương pháp ──────────────────────────────────────────────────
        if method == "auto":
            self.method = "transnet" if _HAS_TRANSNET else "histogram"
        else:
            self.method = method

        self._transnet_model = None

        if self.method == "transnet":
            if not _HAS_TRANSNET:
                print("⚠️  transnetv2-pytorch chưa cài. Tự động chuyển sang Histogram.")
                self.method = "histogram"
            else:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"⏳ Đang tải TransNetV2 lên {self._device}...")
                self._transnet_model = _TransNetV2()
                if self._device == "cuda":
                    self._transnet_model.cuda()
                print("✅ TransNetV2 sẵn sàng.")

        print(f"🔧 KeyframeExtractor khởi tạo với phương pháp: [{self.method.upper()}]")
        print(f"   min_shot_length={self.min_shot_length}  max_kf/shot={self.max_keyframes_per_shot}")

    # ──────────────────────────────────────────────────────────────────────────
    def extract_keyframes(self, video_path: str, output_dir: str) -> int:
        """
        Trích xuất keyframes từ video_path, lưu vào output_dir.

        Returns:
            Số ảnh đã lưu thành công.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        print(f"  🎬 [{self.method.upper()}] Phân tích: {video_name}")

        if self.method == "transnet":
            scenes = self._detect_scenes_transnet(video_path)
        else:
            scenes = self._detect_scenes_histogram(video_path)

        if not scenes:
            print(f"  ⚠️  Không phát hiện được cảnh nào trong {video_name}")
            return 0

        print(f"  📊 Phát hiện {len(scenes)} cảnh (shot). Bắt đầu lưu keyframes...")
        saved = self._save_keyframes(video_path, video_name, scenes, output_dir)
        return saved

    # ══════════════════════════════════════════════════════════════════════════
    # PHƯƠNG PHÁP 1: TransNetV2
    # ══════════════════════════════════════════════════════════════════════════
    def _detect_scenes_transnet(self, video_path: str) -> list:
        """Dùng TransNetV2 để phát hiện scene boundary. Trả về list (start, end)."""
        cap = cv2.VideoCapture(video_path)
        frames_small = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_resized = cv2.resize(frame, (48, 27))
            frame_rgb     = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames_small.append(frame_rgb)
        cap.release()

        if not frames_small:
            return []

        frames_np     = np.array(frames_small, dtype=np.uint8)
        frames_tensor = torch.from_numpy(frames_np)

        BATCH = 500
        p_list = []
        with torch.no_grad():
            for i in range(0, len(frames_tensor), BATCH):
                batch = frames_tensor[i : i + BATCH].to(self._device)
                p_single, _ = self._transnet_model.predict_frames(batch)
                p_list.append(p_single.cpu().numpy())
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        p_single = np.concatenate(p_list, axis=0)
        scenes   = self._transnet_model.predictions_to_scenes(p_single)

        # Lọc shot quá ngắn
        return [
            (int(s), int(e)) for s, e in scenes
            if (e - s + 1) >= self.min_shot_length
        ]

    # ══════════════════════════════════════════════════════════════════════════
    # PHƯƠNG PHÁP 2: Histogram Difference (Fallback)
    # ══════════════════════════════════════════════════════════════════════════
    def _detect_scenes_histogram(self, video_path: str) -> list:
        """
        Phát hiện scene cut bằng chênh lệch histogram HSV giữa các frame liên tiếp.
        Hoàn toàn chỉ dùng OpenCV — cực kỳ nhẹ, không cần GPU.
        """
        cap      = cv2.VideoCapture(video_path)
        scenes   = []
        prev_hist = None
        frame_idx = 0
        scene_start = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Tính histogram HSV
            hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)

            if prev_hist is not None:
                diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
                if diff > self.hist_threshold:
                    # Scene cut phát hiện tại frame_idx
                    if (frame_idx - scene_start) >= self.min_shot_length:
                        scenes.append((scene_start, frame_idx - 1))
                    scene_start = frame_idx

            prev_hist = hist
            frame_idx += 1

        # Thêm cảnh cuối
        if (frame_idx - scene_start) >= self.min_shot_length:
            scenes.append((scene_start, frame_idx - 1))

        cap.release()
        return scenes

    # ══════════════════════════════════════════════════════════════════════════
    # LƯU KEYFRAMES RA DISK
    # ══════════════════════════════════════════════════════════════════════════
    def _save_keyframes(
        self,
        video_path:  str,
        video_name:  str,
        scenes:      list,
        output_dir:  str,
    ) -> int:
        """
        Với mỗi shot (start, end), chọn đều N frame đại diện và lưu .jpg.
        N = min(max_keyframes_per_shot, shot_length // 30)
        """
        cap        = cv2.VideoCapture(video_path)
        saved      = 0
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]

        for scene_start, scene_end in scenes:
            shot_len = scene_end - scene_start + 1

            # Số keyframe cần lấy từ shot này
            n_kf = max(1, min(self.max_keyframes_per_shot, shot_len // 30))

            # Chọn vị trí đều nhau trong shot
            if n_kf == 1:
                positions = [scene_start + shot_len // 2]
            else:
                step = shot_len // (n_kf + 1)
                positions = [scene_start + step * (j + 1) for j in range(n_kf)]

            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if not ret:
                    continue

                fname    = f"{video_name}_frame{pos:06d}.jpg"
                out_path = os.path.join(output_dir, fname)

                # Tiền xử lý (Ép cân, viền đen)
                processed_frame = preprocess_frame(frame)

                if cv2.imwrite(out_path, processed_frame, encode_params):
                    saved += 1

        cap.release()
        return saved


# ── Test nhanh ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Cách dùng: python keyframe_extractor.py <video.mp4> <output_dir>")
        sys.exit(1)

    video  = sys.argv[1]
    outdir = sys.argv[2]
    extractor = KeyframeExtractor(method="auto")
    n = extractor.extract_keyframes(video, outdir)
    print(f"✅ Đã lưu {n} keyframes vào {outdir}")
