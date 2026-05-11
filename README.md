# 🔍 Hệ thống Tìm kiếm Video (CBVR) - Late Fusion & U^2-Net

Hệ thống Tìm kiếm Video dựa trên Nội dung (Content-Based Video Retrieval) tối ưu cho phần cứng giới hạn (8GB RAM). Hệ thống kết hợp 4 đặc trưng đa phương thức và kỹ thuật bóc tách nền bằng AI (U^2-Net) để đạt độ chính xác cao nhất trên tập dữ liệu Động vật.

## ✨ Sơ đồ khối Kiến trúc Hệ thống

*(Nếu bạn đang xem trên GitHub, sơ đồ dưới đây sẽ tự động hiển thị rất trực quan)*

```mermaid
flowchart TD
    %% --- KHAI BÁO CÁC CỤM (SUBGRAPHS) ---
    subgraph Phase1 ["Phase 1 & 1.5: Xử lý Dữ liệu thô (Data Pipeline)"]
        direction TB
        V[Kho Video Gốc .mp4] -->|TransNetV2| SBD[Cắt Cảnh/Shot Boundary Detection]
        SBD --> KFE[Trích xuất Keyframes]
        KFE -->|Resize & Padding| F1[(data/frames/\nẢnh 512x512)]
        F1 -->|rembg| U2[Trí tuệ nhân tạo U^2-Net\nBóc tách Background]
        U2 --> M1[(data/masks/\nẢnh nền đen sắc nét)]
    end

    subgraph Phase2 ["Phase 2: Xây dựng CSDL (Database Building)"]
        direction TB
        M1 --> FE1{Feature Extractor\nLoại bỏ hoàn toàn điểm ảnh Đen}
        
        FE1 -->|ViT-B/32| V1[CLIP Vector\nNgữ nghĩa 512-D]
        FE1 -->|Masked Hist| V2[Color Vector\nMàu sắc HSV 512-D]
        FE1 -->|Grayscale| V3[HOG Vector\nHình dáng 3780-D]
        FE1 -->|Masked Uniform| V4[Texture LBP Vector\nKết cấu Lông/Vằn 26-D]
        
        V1 --> F_CLIP[(FAISS CLIP Index)]
        V2 --> F_COL[(FAISS Color Index)]
        V3 --> F_HOG[(FAISS HOG Index)]
        V4 --> F_TEX[(FAISS Texture Index)]
        
        FE1 --> CSV[(mapping.csv\nLưu Metadata Video)]
    end

    subgraph Phase3 ["Phase 3: Giao diện Streamlit & Tìm kiếm (Retrieval)"]
        direction TB
        UI_IN[Người dùng Upload\nẢnh Truy Vấn] --> U2_QUERY[U^2-Net Tiền Xử Lý\nXóa Background Đồng bộ]
        U2_QUERY --> FE2{Feature Extractor}
        
        FE2 --> Q_CLIP[Q_CLIP]
        FE2 --> Q_COL[Q_Color]
        FE2 --> Q_HOG[Q_HOG]
        FE2 --> Q_TEX[Q_Texture]
        
        Q_CLIP -->|Inner Product| F_CLIP
        Q_COL -->|Inner Product| F_COL
        Q_HOG -->|Inner Product| F_HOG
        Q_TEX -->|Inner Product| F_TEX
        
        F_CLIP --> S1[Scores CLIP]
        F_COL --> S2[Scores Color]
        F_HOG --> S3[Scores HOG]
        F_TEX --> S4[Scores Texture]
        
        S1 & S2 & S3 & S4 --> LF((Toán học\nLATE FUSION))
        LF --> |w1, w2, w3, w4| TOTAL[Total Scores Mọi Frame]
        
        TOTAL --> CSV
        CSV --> GRP[Video Grouping\nLấy Frame Điểm Cao Nhất Của Từng Video]
        GRP --> SORT[Sắp xếp Giảm Dần\nLấy Top 5]
        SORT --> UI_OUT[Hiển thị kết quả & Điểm thành phần]
    end

    %% --- LIÊN KẾT GIỮA CÁC SUBGRAPH ---
    Phase1 ~~~ Phase2
    Phase2 ~~~ Phase3

    %% --- STYLING MÀU SẮC CHO ĐẸP ---
    style Phase1 fill:#f9f2ec,stroke:#e67e22,stroke-width:2px,color:#d35400
    style Phase2 fill:#eaf2f8,stroke:#2980b9,stroke-width:2px,color:#154360
    style Phase3 fill:#e8f8f5,stroke:#1abc9c,stroke-width:2px,color:#0e6251
    style LF fill:#f1c40f,stroke:#f39c12,stroke-width:4px,color:#000
    style FE1 fill:#9b59b6,stroke:#8e44ad,color:#fff
    style FE2 fill:#9b59b6,stroke:#8e44ad,color:#fff
    style U2 fill:#e74c3c,stroke:#c0392b,color:#fff
    style U2_QUERY fill:#e74c3c,stroke:#c0392b,color:#fff
```

## 🚀 Cách thức hoạt động
1. **Phase 1 (`run_phase1_extract_frames.py`)**: Dùng `TransNetV2` để phát hiện chuyển cảnh (Shot Boundary) và trích xuất Keyframe chuẩn 512x512 có viền đen.
2. **Phase 1.5 (`run_phase1_5_smart_crop.py`)**: Dùng mạng nơ-ron `U^2-Net` (`rembg`) để bóc tách nền (background) cực kỳ sắc nét, giữ lại hoàn toàn "thịt" động vật.
3. **Phase 2 (`run_phase2_build_index.py`)**: Đọc ảnh đã xóa nền, sử dụng Mask để **bỏ qua nền đen**, tính toán 4 Vector đặc trưng (CLIP, Color, HOG, LBP) và nạp vào CSDL FAISS.
4. **Phase 3 (`app.py`)**: Giao diện Streamlit cho phép upload ảnh. Ảnh cũng được đưa qua `U^2-Net` để đảm bảo tính đồng nhất. Sau đó thuật toán `Late Fusion` và `Video Grouping` sẽ xếp hạng và trả về 5 Video liên quan nhất.

## 2.1. Ti?n x? l�
To�n b? video nh�m s? d?ng du?c chu?n h�a v? d?nh d?ng th?ng nh?t:
� �?c video t? thu m?c d?u v�o, h? tr? c�c d?nh d?ng .mp4, .avi, .mov, .mkv, .webm
� Tr�ch xu?t khung h�nh (Keyframe): Video du?c chia th�nh c�c shot theo nguy�n l� ph�t hi?n thay d?i c?nh th�ng qua hai phuong ph�p:
  - **Phuong ph�p 1 (Uu ti�n) - M?ng no-ron TransNetV2**: M?ng h?c s�u d? do�n di?m c?t c?nh v?i d? ch�nh x�c cao tr�n t?ng batch (500 frames/l?n).
  - **Phuong ph�p 2 (D? ph�ng) - Bi?u d? m�u (Histogram Difference)**: H? th?ng t? d?ng chuy?n sang so s�nh kho?ng c�ch Bhattacharyya gi?a c�c histogram HSV li�n ti?p. N?u d? kh�c bi?t > 0.4, m?t shot m?i du?c d�nh d?u.
� L?c v� l?y m?u: B? qua c�c shot qu� ng?n (du?i 15 frame). M?i shot h?p l? s? l?y d?i di?n t?i da 3 keyframes ? c�c v? tr� chia d?u.
� Ti?n x? l� ?nh (Letterboxing): T?t c? c�c keyframe du?c thu nh? sao cho l?n nh?t kh�ng vu?t qu� 512 pixel, gi? nguy�n t? l? khung h�nh th?t. Ph?n b? du du?c d?n vi?n den d? t?o ra ?nh vu�ng chu?n 512x512 gi�p t?i uu h�a khi dua v�o c�c m?ng h?c s�u.
� X�a ph�ng (Smart Cropping): To�n b? ?nh keyframe 512x512 du?c ch?y qua m� h�nh **U^2-Net** d? t�ch n?n t? d?ng. Ph?n c?nh quan du?c bi?n th�nh m�u den tuy?t d?i (pixel=0), tri?t ti�u ho�n to�n s? nhi?u lo?n m�u s?c.
