from pathlib import Path
from PIL import Image

# Change this to your image directory
IMAGE_DIR = Path(r"./captures/session_05_20260624/images")

# Find all matching images recursively
for image_path in IMAGE_DIR.rglob("*_right_fisheye_image.jpg"):
    try:
        with Image.open(image_path) as img:
            # Rotate 180° (lossless for JPEG dimensions)
            rotated = img.rotate(180, expand=False)

            # Overwrite the original image
            rotated.save(image_path)

        print(f"Rotated: {image_path}")

    except Exception as e:
        print(f"Failed: {image_path} ({e})")

print("Done.")