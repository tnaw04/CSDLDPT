"""
PHASE 2 — FeatureExtractor
============================
Nhiệm vụ: Đọc ảnh .jpg keyframe → Trích xuất vector đặc trưng.

Hai luồng đặc trưng:
    - Luồng 1 (Ngữ nghĩa): CLIP ViT-B/32 → vector 512 chiều
    - Luồng 2 (Màu sắc) : Color Histogram HSV → vector 512 chiều
    - Luồng 3 (Hình dáng) : HOG Descriptor → vector 3780 chiều (sau đó có thể tuỳ chọn PCA/giảm chiều)

Tất cả vector được chuẩn hóa L2 trước khi trả về.
Thư viện: clip, torch, cv2, numpy, PIL
"""

import numpy as np
import cv2
import torch
import clip
from PIL import Image
from skimage.feature import local_binary_pattern


class FeatureExtractor:
    """
    Trích xuất đặc trưng từ ảnh keyframe phục vụ Phase 2.

    Lưu ý: Chỉ khởi tạo class này trong Phase 2.
    Toàn bộ RAM trong Phase 2 sẽ dành riêng cho CLIP và PostgreSQL.
    """

    # Bins cho Color Histogram HSV: H=16, S=8, V=4 → 16*8*4 = 512 chiều
    _H_BINS = 16
    _S_BINS = 8
    _V_BINS = 4

    def __init__(self, device: str | None = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"⏳ Khởi tạo FeatureExtractor (CLIP ViT-B/32) trên [{self.device}]...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()
        print("✅ FeatureExtractor sẵn sàng.")

    # ──────────────────────────────────────────────────────────────────────────
    # Luồng 1: CLIP Semantic Vector (512-D)
    # ──────────────────────────────────────────────────────────────────────────
    def extract_clip_vector(self, image_path: str) -> np.ndarray:
        """Trích xuất vector ngữ nghĩa 512 chiều từ ảnh bằng CLIP."""
        img        = Image.open(image_path).convert("RGB")
        img_tensor = self.clip_preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.clip_model.encode_image(img_tensor)

        vec  = feat.cpu().numpy().flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm   # L2 normalize
        return vec

    # ──────────────────────────────────────────────────────────────────────────
    # Luồng 2: Color Histogram HSV (512-D)
    # ──────────────────────────────────────────────────────────────────────────
    def extract_color_histogram(self, image_path):
        """
        Trích xuất đặc trưng Color Histogram 3D (HSV hoặc RGB) của con vật.
        ĐÃ FIX: Bỏ qua nền đen (viền padding và nền tách bởi U^2-Net) để không đếm màu đen.
        """
        image = cv2.imread(image_path)
        if image is None:
            return np.zeros(512, dtype=np.float32)
            
        image = cv2.resize(image, (512, 512))
        
        # 1. Tạo mask: Lấy tất cả các pixel KHÁC màu đen (0,0,0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # 2. Tính Histogram chỉ trên phần mask (phần có thịt con vật)
        hist = cv2.calcHist([image], [0, 1, 2], mask, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    # ──────────────────────────────────────────────────────────────────────────
    # Luồng 3: HOG (Histogram of Oriented Gradients)
    # ──────────────────────────────────────────────────────────────────────────
    def extract_hog_vector(self, image_path: str) -> np.ndarray:
        """Trích xuất vector hình khối 3780 chiều bằng HOG (không cần GPU)."""
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Không đọc được ảnh: {image_path}")
            
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(img_gray, (64, 128)) # Standard size cho HOG mặc định
        
        # Khởi tạo HOG Descriptor mặc định
        hog = cv2.HOGDescriptor()
        
        hist = hog.compute(img_resized)
        vec  = hist.flatten().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm   # L2 normalize
        return vec

    # ──────────────────────────────────────────────────────────────────────────
    # Luồng 4: Kết cấu bề mặt (Texture LBP)
    # ──────────────────────────────────────────────────────────────────────────
    def extract_texture_lbp(self, image_path, numPoints=24, radius=3):
        """
        Trích xuất đặc trưng Kết cấu (Texture) bằng Local Binary Pattern (LBP).
        ĐÃ FIX: Chỉ đếm các mẫu họa tiết nằm trong vùng con vật (bỏ qua nền đen).
        """
        n_bins = numPoints + 2
        image = cv2.imread(image_path)
        if image is None:
            return np.zeros(n_bins, dtype=np.float32)
            
        image = cv2.resize(image, (512, 512))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Tạo mask loại bỏ nền đen
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        lbp = local_binary_pattern(gray, numPoints, radius, method="uniform")
        
        # Tính histogram của LBP nhưng CHỈ LẤY CÁC PIXEL NẰM TRONG MASK
        # np.histogram nhận mảng 1D, ta filter bằng mask: lbp[mask > 0]
        valid_lbp_pixels = lbp[mask > 0]
        
        if len(valid_lbp_pixels) == 0:
            return np.zeros(n_bins, dtype=np.float32)
            
        hist, _ = np.histogram(valid_lbp_pixels, bins=n_bins, range=(0, n_bins))
        
        # Chuẩn hóa L2-norm
        hist = hist.astype("float")
        eps = 1e-7
        hist /= (hist.sum() + eps)
        
        return hist.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Hàm tổng hợp: trả về (clip_vec, color_vec, hog_vec, texture_vec)
    # ──────────────────────────────────────────────────────────────────────────
    def extract(self, image_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Trích xuất cả 4 vector từ một ảnh.

        Returns:
            (clip_vec, color_vec, hog_vec, texture_vec) — cả 4 đã L2-normalize, dtype float32
        """
        return (
            self.extract_clip_vector(image_path),
            self.extract_color_histogram(image_path),
            self.extract_hog_vector(image_path),
            self.extract_texture_lbp(image_path),
        )
