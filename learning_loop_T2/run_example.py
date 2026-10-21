#!/usr/bin/env python
"""
Example: Run visual clustering on your sign language dataset
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("=" * 70)
    print("VISUAL CLUSTERING - EXAMPLE RUN")
    print("=" * 70)
    
    # Check if input folder exists
    input_folder = Path("data/images")
    if not input_folder.exists():
        print(f"\n❌ Error: {input_folder} not found")
        print("Please ensure your images are in data/images/")
        return 1
    
    # Count images
    image_count = len(list(input_folder.glob("**/*.jpg"))) + \
                  len(list(input_folder.glob("**/*.png"))) + \
                  len(list(input_folder.glob("**/*.jpeg")))
    
    if image_count == 0:
        print(f"\n❌ No images found in {input_folder}")
        return 1
    
    print(f"\n✓ Found {image_count} images in {input_folder}")
    
    # Mode 1: With class names
    print("\n" + "─" * 70)
    print("Running Mode 1: With class names")
    print("─" * 70)
    
    output_mode1 = "sign_output_named"
    class_names = ["two", "three", "four", "six", "seven", "eight", "nine"]
    
    cmd_mode1 = [
        sys.executable,
        "auto_pipeline.py",
        str(input_folder),
        output_mode1
    ] + class_names
    
    print(f"\nCommand: {' '.join(cmd_mode1)}")
    print(f"\nExpected output: {output_mode1}/")
    for name in class_names:
        print(f"  {name}/")
    
    print("\n▶️  Running...")
    result1 = subprocess.run(cmd_mode1, capture_output=False)
    
    if result1.returncode == 0:
        print("\n✅ Mode 1 completed successfully!")
        print(f"Check {output_mode1}/ for results")
    else:
        print("\n❌ Mode 1 failed")
        return 1
    
    # Mode 2: With number only
    print("\n\n" + "─" * 70)
    print("Running Mode 2: With number only")
    print("─" * 70)
    
    output_mode2 = "sign_output_numbered"
    n_classes = "7"
    
    cmd_mode2 = [
        sys.executable,
        "auto_pipeline.py",
        str(input_folder),
        output_mode2,
        n_classes
    ]
    
    print(f"\nCommand: {' '.join(cmd_mode2)}")
    print(f"\nExpected output: {output_mode2}/")
    for i in range(1, int(n_classes) + 1):
        print(f"  class_{i}/")
    
    print("\n▶️  Running...")
    result2 = subprocess.run(cmd_mode2, capture_output=False)
    
    if result2.returncode == 0:
        print("\n✅ Mode 2 completed successfully!")
        print(f"Check {output_mode2}/ for results")
    else:
        print("\n❌ Mode 2 failed")
        return 1
    
    # Final summary
    print("\n" + "=" * 70)
    print("BOTH MODES COMPLETED")
    print("=" * 70)
    print(f"\nMode 1 (named): {output_mode1}/")
    print(f"Mode 2 (numbered): {output_mode2}/")
    print("\nCompare the results to see how clustering works!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
