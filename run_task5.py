from pathlib import Path
from collections import Counter

try:
    dataset_root = Path("Dataset")
    print("=== TASK 5 OUTPUT ===")
    print("=== Per Subfolder ===")
    for folder in sorted(dataset_root.rglob("*")):
        if not folder.is_dir(): continue
        imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
        gts  = list(folder.glob("*_gt.json"))
        if imgs:
            print(f"{folder.name:30s} | {len(imgs):4d} gambar | {len(gts)} GT")
    
    exts = Counter(f.suffix.lower() for f in dataset_root.rglob("*") if f.is_file())
    print("\n=== Distribusi Ekstensi ===")
    for ext, count in exts.most_common():
        print(f"  {ext}: {count}")
except Exception as e:
    print(f"Error in Task 5: {e}")
