from pathlib import Path
import shutil

# --- CONFIG ---
source_dir = Path("./captures/session_05_20260624/images")
target_dir = Path("./frontback_dir")

# Create target directory if it doesn't exist
target_dir.mkdir(parents=True, exist_ok=True)

# Filename patterns to match
patterns = [
    "*_back_fisheye_image.jpg",
    "*_frontright_fisheye_image.jpg",
    "*_frontleft_fisheye_image.jpg",
]

# Collect and copy matching files
copied = 0

for pattern in patterns:
    for file_path in source_dir.glob(pattern):
        if file_path.is_file():
            dest_path = target_dir / file_path.name
            shutil.copy2(file_path, dest_path)
            copied += 1
            print(f"Copied: {file_path.name}")

print(f"\nDone. Total files copied: {copied}")