from PIL import Image
import io
import base64

MAX_DIMENSION = 1568  # OpenAI/Azure vision models downscale beyond this anyway
MAX_FILE_SIZE_BYTES = 4 * 1024 * 1024  # 4MB safety cap per image

def encode_image(image_path: str) -> str:
    try:
        with Image.open(image_path) as img:
            # Force-convert to RGB (handles HEIC-as-jpg, CMYK, palette, RGBA, etc.)
            if img.mode != "RGB":
                img = img.convert("RGB")
            # Downscale if oversized (fixes the 11.9MB payload hang)
            if max(img.size) > MAX_DIMENSION:
                img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            data = buf.getvalue()
            if len(data) > MAX_FILE_SIZE_BYTES:
                # second pass, more aggressive compression
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=60)
                data = buf.getvalue()
            return base64.b64encode(data).decode("utf-8")
    except Exception as e:
        raise ValueError(f"Unreadable image file: {image_path}")
