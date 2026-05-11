import os
import cv2
import numpy as np
from tqdm import tqdm

# pyrefly: ignore [missing-import]
from rembg import remove
from PIL import Image

def ultimate_segment_animal(frame, target_size=(512, 512)):
    """
    Tiền xử lý tối thượng: Dùng U^2-Net (rembg) bóc tách đối tượng sắc lẹm -> Ép viền đen.
    """
    # 1. rembg yêu cầu ảnh đầu vào là RGB (OpenCV mặc định là BGR)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Xóa nền chỉ với 1 DÒNG CODE! (Kết quả trả về là ảnh có nền trong suốt - RGBA)
    try:
        output_rgba = remove(img_rgb)
    except Exception as e:
        print(f"Lỗi tách nền: {e}")
        output_rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) # Fallback
    
    # 3. Tạo nền đen thay cho nền trong suốt
    # Tách kênh alpha (độ trong suốt)
    alpha_channel = output_rgba[:, :, 3]
    
    # Đắp phần thịt của con vật lên nền đen
    mask = (alpha_channel > 50).astype(np.uint8) # Ngưỡng 50 để lọc viền nhiễu
    foreground = cv2.cvtColor(output_rgba, cv2.COLOR_RGBA2BGR) # Chuyển lại về BGR cho OpenCV
    
    # Ở đâu mask = 1 thì lấy con vật, = 0 thì lấy màu đen
    segmented_img = cv2.bitwise_and(foreground, foreground, mask=mask)
    
    # 4. Tìm Bounding Box của con vật từ Mask để cắt cho to ra
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Lấy khung bao trùm lớn nhất
        c = max(contours, key=cv2.contourArea)
        x, y, w, h_box = cv2.boundingRect(c)
        
        # Mở rộng 5%
        h, w_img = frame.shape[:2]
        mx, my = int(w * 0.05), int(h_box * 0.05)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(w_img, x + w + mx), min(h, y + h_box + my)
        
        cropped_img = segmented_img[y1:y2, x1:x2]
    else:
        cropped_img = segmented_img

    # 5. Ép cân về 512x512 (Độn viền đen)
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

def main():
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    FRAMES_DIR = os.path.join(PROJECT_ROOT, "data", "frames")
    MASKS_DIR = os.path.join(PROJECT_ROOT, "data", "masks")
    
    os.makedirs(MASKS_DIR, exist_ok=True)
    
    # Lấy danh sách tất cả ảnh đã cắt (từ data/frames)
    image_files = [f for f in os.listdir(FRAMES_DIR) if f.endswith(('.jpg', '.png'))]
    print(f"🔍 Tìm thấy {len(image_files)} ảnh cần bóc nền bằng U^2-Net.")
    if len(image_files) == 0:
        return
        
    print("⏳ Lần chạy đầu tiên sẽ tốn khoảng vài chục giây để tải Model u2net.onnx...")
    
    # Quét qua từng ảnh
    success_cnt = 0
    for img_name in tqdm(image_files, desc="Đang tách nền"):
        img_path = os.path.join(FRAMES_DIR, img_name)
        mask_path = os.path.join(MASKS_DIR, img_name)
        
        # 1. Đọc ảnh cũ lên
        img = cv2.imread(img_path)
        if img is None: 
            continue
            
        # 2. Xử lý qua rembg
        try:
            processed_img = ultimate_segment_animal(img)
            
            # 3. GHI FILE MỚI VÀO THƯ MỤC MASKS
            cv2.imwrite(mask_path, processed_img)
            success_cnt += 1
        except Exception as e:
            print(f"Lỗi ảnh {img_name}: {e}")
            
    print(f"✅ Hoàn tất U^2-Net Segmentation cho {success_cnt}/{len(image_files)} ảnh!")
    print(f"📂 Ảnh nền đen tuyệt đẹp đã được lưu vào thư mục: {MASKS_DIR}")
    print("👉 Bây giờ bạn chạy lại Phase 2 (run_phase2_build_index.py) để nạp.")

if __name__ == "__main__":
    main()
