# VieNeu local cho gen-video

Adapter ở `apps/gen_video_api.py` dùng SDK của repo (3.8.3), v3 Turbo ONNX/CPU
fp32, hỗ trợ giọng preset và clone. Không cần NVIDIA GPU/PyTorch.

## Docker

Chạy các lệnh dưới đây tại thư mục `VieNeu-TTS`. Network
`gen-video_gen_video_network` cần tồn tại (được tạo khi chạy stack gen-video).
Nếu tên network khác, đặt `GEN_VIDEO_NETWORK` trước khi chạy Compose.

Lần đầu trên máy chưa có model/cache:

```powershell
$env:VIENEU_OFFLINE = '0'
docker compose -f docker/compose.gen-video.yml up -d --build
docker logs -f gen_video_vieneu
```

Đợi log `Local preset + cloning engine ready` và trạng thái healthy. Startup
warm-up cả preset và clone để tải đủ codec, speaker encoder và denoiser.

Sau khi tải đủ, bật lại chế độ offline (mặc định của Compose):

```powershell
$env:VIENEU_OFFLINE = '1'
docker compose -f docker/compose.gen-video.yml up -d
```

Khởi động các lần sau với cache đã có:

```powershell
docker compose -f docker/compose.gen-video.yml up -d
```

## Địa chỉ và tài nguyên

- Container: `gen_video_vieneu`, image: `vieneu-gen-video:cpu`.
- API/docs trên máy: http://127.0.0.1:17860/docs
- Health: http://127.0.0.1:17860/health
- API trong network gen-video: `http://gen_video_vieneu:8000`.
- Volume giữ model/cache: `vieneu_local_cache`.
- Giới hạn: 6 CPU, 8 GB RAM; một tác vụ inference tại một thời điểm.
- Lấy danh sách giọng từ `GET /voices`; SDK là nguồn tên giọng hiện hành.
- `POST /synthesize` nhận `text`, `voice` hoặc `reference_audio` (audio data URL),
  `temperature`, `max_new_frames`; trả WAV PCM16 mono 48 kHz.
- Audio WebM/Opus, M4A, MP3 được FFmpeg chuyển sang WAV trước khi clone.
- API local chỉ bind loopback trên máy, backend gọi trong network Docker.

## Kết nối gen-video

`gen-video/docker-compose.override.yml` cấu hình backend:

```text
VIENEU_LOCAL_URL=http://gen_video_vieneu:8000
```

Trong thư mục gen-video, chạy `docker compose up -d --build client server` sau khi
VieNeu healthy. Khi biến trên được đặt, backend dùng local cho cả preset và clone;
không fallback sang Hugging Face nếu local lỗi. Khi chạy NestJS trực tiếp trên máy
(không trong Docker), đặt URL thành `http://127.0.0.1:17860`.

## Lưu bền vững và kiểm tra offline

Rebuild image hoặc recreate container không xoá model. Không dùng `down -v` nếu
muốn giữ cache: cờ đó xoá volume. Sao lưu volume `vieneu_local_cache` và image
`vieneu-gen-video:cpu` để có thể khôi phục trên máy khác không cần tải lại.

Đã kiểm tra tạo preset và clone trong container `--network none` với
`HF_HUB_OFFLINE=1`, dùng cache này. Chế độ offline chỉ áp dụng engine VieNeu;
đăng nhập và các chức năng provider AI khác của gen-video vẫn có yêu cầu mạng riêng.
