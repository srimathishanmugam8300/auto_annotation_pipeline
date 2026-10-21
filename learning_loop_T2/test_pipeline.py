"""
Quick test script for the visual clustering pipeline
"""

import sys
from pathlib import Path

# Test both modes

print("=" * 60)
print("TESTING VISUAL CLUSTERING PIPELINE")
print("=" * 60)

# Check if images exist
input_folder = Path("data/images")
if not input_folder.exists():
    print(f"\n❌ Error: {input_folder} not found")
    print("Please create a folder with images first")
    sys.exit(1)

# Count images
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
images = []
for ext in image_extensions:
    images.extend(input_folder.glob(f"**/*{ext}"))
    images.extend(input_folder.glob(f"**/*{ext.upper()}"))

if len(images) == 0:
    print(f"\n❌ No images found in {input_folder}")
    sys.exit(1)

print(f"\n✓ Found {len(images)} images in {input_folder}")

print("\n" + "─" * 60)
print("MODE 1: Test with class names")
print("─" * 60)
print("\nCommand:")
print("  python auto_pipeline.py data/images output_mode1 two three four six seven eight nine")
print("\nExpected Output:")
print("  output_mode1/")
print("    two/")
print("    three/")
print("    four/")
print("    six/")
print("    seven/")
print("    eight/")
print("    nine/")

print("\n" + "─" * 60)
print("MODE 2: Test with number only")
print("─" * 60)
print("\nCommand:")
print("  python auto_pipeline.py data/images output_mode2 7")
print("\nExpected Output:")
print("  output_mode2/")
print("    class_1/")
print("    class_2/")
print("    class_3/")
print("    class_4/")
print("    class_5/")
print("    class_6/")
print("    class_7/")

print("\n" + "=" * 60)
print("Ready to test! Run one of the commands above.")
print("=" * 60)
