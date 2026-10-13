"""
Fast Multi-Threaded Annotation & Overlay Scanner for BoneGuard AI
Detects red arrows, colored arrows, circles, boxes, lines, and drawn annotations.
"""
import os
import glob
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

base_dir = r"D:\BoneGuard AI"

def check_single_image(img_path):
    try:
        with Image.open(img_path) as pil_img:
            # Downscale slightly for rapid scan while keeping any annotations intact
            w, h = pil_img.size
            if max(w, h) > 512:
                scale = 512.0 / max(w, h)
                pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.Resampling.NEAREST)
            img_rgb = np.array(pil_img.convert('RGB'))
    except Exception as e:
        return {'path': img_path, 'filename': os.path.basename(img_path), 'error': str(e), 'has_color_annotation': False}

    h, w, c = img_rgb.shape
    r = img_rgb[:, :, 0].astype(np.int16)
    g = img_rgb[:, :, 1].astype(np.int16)
    b = img_rgb[:, :, 2].astype(np.int16)
    
    diff_rg = np.abs(r - g)
    diff_rb = np.abs(r - b)
    diff_gb = np.abs(g - b)
    max_diff = np.maximum(np.maximum(diff_rg, diff_rb), diff_gb)
    
    colored_pixels = np.sum(max_diff > 30)
    colored_ratio = colored_pixels / (h * w)
    
    # Red annotations (high red, low green and blue)
    red_mask = (r > 130) & (g < 110) & (b < 110) & ((r - g) > 35) & ((r - b) > 35)
    red_pixels = int(np.sum(red_mask))
    
    # Yellow annotations (high red and green, low blue)
    yellow_mask = (r > 130) & (g > 130) & (b < 100) & (np.abs(r - g) < 50) & ((r - b) > 40)
    yellow_pixels = int(np.sum(yellow_mask))
    
    # Blue / Cyan annotations
    blue_mask = (b > 130) & (r < 110) & ((b - r) > 35)
    blue_pixels = int(np.sum(blue_mask))
    
    # Green annotations
    green_mask = (g > 130) & (r < 110) & (b < 110) & ((g - r) > 35) & ((g - b) > 35)
    green_pixels = int(np.sum(green_mask))

    has_color_annotation = False
    color_type = 'NONE'
    
    if red_pixels > 15:
        has_color_annotation = True
        color_type = 'RED_ARROW_OR_MARKING'
    elif yellow_pixels > 15:
        has_color_annotation = True
        color_type = 'YELLOW_MARKING'
    elif blue_pixels > 15:
        has_color_annotation = True
        color_type = 'BLUE_MARKING'
    elif green_pixels > 15:
        has_color_annotation = True
        color_type = 'GREEN_MARKING'
    elif colored_pixels > 80 and colored_ratio > 0.0002:
        has_color_annotation = True
        color_type = 'GENERIC_COLORED_OVERLAY'

    return {
        'path': img_path,
        'filename': os.path.basename(img_path),
        'has_color_annotation': has_color_annotation,
        'color_type': color_type,
        'red_pixels': red_pixels,
        'yellow_pixels': yellow_pixels,
        'blue_pixels': blue_pixels,
        'green_pixels': green_pixels,
        'colored_pixels': int(colored_pixels),
        'colored_ratio': float(colored_ratio)
    }

def scan_file_list(file_list, num_workers=12):
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for res in tqdm(executor.map(check_single_image, file_list), total=len(file_list)):
            results.append(res)
    return results

print("1. Scanning Raw BFRMX dataset...")
bfrmx_files = glob.glob(os.path.join(base_dir, "01_Raw_Datasets", "BFRMX", "**", "*.*"), recursive=True)
bfrmx_files = [f for f in bfrmx_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
bf_results = scan_file_list(bfrmx_files)
df_bf_ann = pd.DataFrame(bf_results)
df_bf_ann['class'] = df_bf_ann['path'].apply(lambda f: 'FRACTURE' if 'fractured' in f.lower() and 'not fractured' not in f.lower() else 'NO_FRACTURE')
df_bf_ann['dataset'] = 'BFRMX'

print("\n2. Scanning Raw FracAtlas dataset...")
fa_files = glob.glob(os.path.join(base_dir, "01_Raw_Datasets", "FracAtlas", "images", "**", "*.*"), recursive=True)
fa_files = [f for f in fa_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
fa_results = scan_file_list(fa_files)
df_fa_ann = pd.DataFrame(fa_results)
df_fa_ann['class'] = df_fa_ann['path'].apply(lambda f: 'FRACTURE' if 'fractured' in f.lower() and 'non_fractured' not in f.lower() else 'NO_FRACTURE')
df_fa_ann['dataset'] = 'FracAtlas'

print("\n3. Scanning 04_Preprocessing/02_Split (Train/Validation/Test)...")
split_files = glob.glob(os.path.join(base_dir, "04_Preprocessing", "02_Split", "**", "*.*"), recursive=True)
split_files = [f for f in split_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
split_results = scan_file_list(split_files)
df_split_ann = pd.DataFrame(split_results)
df_split_ann['split'] = df_split_ann['path'].apply(lambda f: "train" if "\\train\\" in f else ("validation" if "\\validation\\" in f else "test"))
df_split_ann['class'] = df_split_ann['path'].apply(lambda f: "FRACTURE" if "\\FRACTURE\\" in f else "NO_FRACTURE")

print("\n" + "="*70)
print("=== ANNOTATION SCAN RESULTS ===")
print("="*70)

print(f"\n[BFRMX RAW DATASET] Total Images: {len(df_bf_ann):,}")
bf_annotated = df_bf_ann[df_bf_ann['has_color_annotation']]
print(f"Manually Drawn / Colored Annotations: {len(bf_annotated):,} ({len(bf_annotated)/len(df_bf_ann)*100:.2f}%)")
if len(bf_annotated) > 0:
    print(bf_annotated[['class', 'color_type']].value_counts())
    print("Sample BFRMX annotated paths:")
    for p in bf_annotated['path'].head(10):
        print("  -", p)

print(f"\n[FRACATLAS RAW DATASET] Total Images: {len(df_fa_ann):,}")
fa_annotated = df_fa_ann[df_fa_ann['has_color_annotation']]
print(f"Manually Drawn / Colored Annotations: {len(fa_annotated):,} ({len(fa_annotated)/len(df_fa_ann)*100:.2f}%)")
if len(fa_annotated) > 0:
    print(fa_annotated[['class', 'color_type']].value_counts())
    print("Sample FracAtlas annotated paths:")
    for p in fa_annotated['path'].head(10):
        print("  -", p)

print(f"\n[04_PREPROCESSING/02_SPLIT] Total Images in Splits: {len(df_split_ann):,}")
split_annotated = df_split_ann[df_split_ann['has_color_annotation']]
print(f"Manually Drawn / Colored Annotations in Split Folders: {len(split_annotated):,}")
if len(split_annotated) > 0:
    print(split_annotated.groupby(['split', 'class', 'color_type']).size())
    print("Sample split annotated paths:")
    for p in split_annotated['path'].head(10):
        print("  -", p)
