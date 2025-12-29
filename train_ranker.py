import pickle
import os
import glob
import random
import numpy as np
from sklearn.linear_model import LogisticRegression

# Import the feature extractors
from src.utils.clip_engine import clip_engine
from src.utils.feature_extraction import (
    calculate_coherence_score,
    calculate_color_harmony,
    calculate_visual_balance
)

# Configuration
MODEL_SAVE_PATH = "models/ranker_weights.pkl"
N_SAMPLES = 100  # Total training samples needed
# Low number bc my pc slow

def load_real_wardrobe_items():
    """
    Scans data/user/items/ recursively to find images.
    Handles structure: data/user/items/{Gender}/{Category}/*.jpg
    """
    # Categories
    wardrobe = {'top': [], 'bottom': [], 'shoes': [], 'dress': []}
    
    # The root folder for items
    base_path = "data/user/items"
    
    print("Scanning for images in 'data/user/items/'")
    
    total_found = 0
    
    # Go through all subdirectories (Male, Female, Unisex, etc.)
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                parent_folder = os.path.basename(root).lower() # e.g., 'top', 'bottom', 'dress'
                
                # Check which category this folder belongs to
                if 'top' in parent_folder:
                    wardrobe['top'].append(full_path)
                elif 'bottom' in parent_folder or 'pant' in parent_folder or 'skirt' in parent_folder:
                    wardrobe['bottom'].append(full_path)
                elif 'shoe' in parent_folder or 'footwear' in parent_folder:
                    wardrobe['shoes'].append(full_path)
                elif 'dress' in parent_folder:
                    wardrobe['dress'].append(full_path)
                
    # Summary
    for cat, items in wardrobe.items():
        if items:
            print(f"   - Found {len(items)} {cat}s (combined)")
            total_found += len(items)
            
    return wardrobe, total_found

def extract_features_from_real_outfit(item_paths):
    """
    Extract features on a list of items (works for 2 items or 3 items).
    item_paths: [top_path, bot_path, shoe_path] OR [dress_path, shoe_path]
    """
    # Get Embeddings
    embeddings = []
    for path in item_paths:
        embeddings.append(clip_engine.get_image_embedding(path))
    
    # Compute Features
    
    # Coherence
    coherence = calculate_coherence_score(embeddings)
    
    # Color Harmony
    color = calculate_color_harmony(item_paths)
    
    # Visual Balance
    balance = calculate_visual_balance(item_paths)
    
    # Style Fit (Simulated)
    style_fit = random.uniform(0.3, 0.95) 
    
    return [style_fit, coherence, color, balance]

def generate_training_data(n_samples):
    """
    Generates dataset X using a mix of REAL data and SYNTHETIC data.
    Handles both Standard Outfits (3 items) and Dress Outfits (2 items).
    """
    X_features = []
    
    # Try to generate from Real Images
    wardrobe, total_items = load_real_wardrobe_items()
    
    real_samples_count = 0
    if total_items > 5:
        print(f"⚡ Generating features from REAL data...")
        
        max_real_tries = int(n_samples * 0.6) 
        
        for _ in range(max_real_tries):
            # Randomize where outfits contains dress or not
            use_dress = False
            if wardrobe['dress'] and random.random() < 0.3: # 30%
                use_dress = True
            
            try:
                outfit_paths = []
                
                if use_dress and wardrobe['dress'] and wardrobe['shoes']:
                    # Build Dress Outfit
                    d = random.choice(wardrobe['dress'])
                    s = random.choice(wardrobe['shoes'])
                    outfit_paths = [d, s]
                    
                elif wardrobe['top'] and wardrobe['bottom'] and wardrobe['shoes']:
                    # Build Standard Outfit
                    t = random.choice(wardrobe['top'])
                    b = random.choice(wardrobe['bottom'])
                    s = random.choice(wardrobe['shoes'])
                    outfit_paths = [t, b, s]
                else:
                    # Not enough items for either type
                    continue

                # Extract features
                feats = extract_features_from_real_outfit(outfit_paths)
                X_features.append(feats)
                real_samples_count += 1
                
                if real_samples_count % 10 == 0:
                    print(f"    processed {real_samples_count} real outfits", end="\r")
                    
            except Exception as e:
                print(f"Skip error: {e}")
                continue
                
        print(f"\nGenerated {real_samples_count} vectors from real images")
    
    # Fill the rest with Synthetic Data
    remaining = n_samples - len(X_features)
    if remaining > 0:
        print(f"Synthesizing {remaining} additional samples")
        
        # Synthetic generation stays the same (it mocks the Feature Vector, not the items)
        style_fits = np.random.normal(0.5, 0.25, remaining)
        coherences = np.random.normal(0.45, 0.2, remaining) 
        colors = np.random.normal(0.5, 0.3, remaining)
        balances = np.random.normal(0.6, 0.2, remaining)
        
        synthetic_chunk = np.column_stack([style_fits, coherences, colors, balances])
        for row in synthetic_chunk:
            X_features.append(row)
            
    X_final = np.array(X_features)
    X_final = np.clip(X_final, 0.0, 1.0)
    
    return X_final

def expert_labeling_teacher(features):
    """
    Perform pseudo labeling
    """
    style, coh, color, bal = features
    
    # Case 1: Good Outfit
    if style > 0.70 and coh > 0.60 and color > 0.55:
        return 1
    
    # Case 2: Bad Outfit
    elif style < 0.45 or color < 0.35 or bal < 0.3:
        return 0
        
    # Case 3: Ambiguous
    else:
        return -1

def train_ranker():
    # Generate Data (Hybrid: Real + Synthetic)
    X_raw = generate_training_data(N_SAMPLES)
    
    X_train = []
    y_train = []
    
    # Apply Self-Supervision
    print("Teacher is labeling the data")
    count_good = 0
    count_bad = 0
    
    for row in X_raw:
        label = expert_labeling_teacher(row)
        if label != -1:
            X_train.append(row)
            y_train.append(label)
            if label == 1: count_good += 1
            else: count_bad += 1
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"    Data labeled. Good: {count_good}, Bad: {count_bad}, Total: {len(X_train)}")

    if len(X_train) == 0:
        print("Error: No valid training samples found")
        return

    # Train Model
    print("Training Logistic Regression Model")
    clf = LogisticRegression(fit_intercept=True, solver='lbfgs')
    clf.fit(X_train, y_train)
    
    # Extract Weights
    weights = clf.coef_[0]
    intercept = clf.intercept_[0]
    
    print("\n" + "="*40)
    print("   TRAINING COMPLETE")
    print("   The model has learned from your data:")
    print(f"   1. Style Fit:      {weights[0]:.4f}")
    print(f"   2. Coherence:      {weights[1]:.4f}")
    print(f"   3. Color Harmony:  {weights[2]:.4f}")
    print(f"   4. Visual Balance: {weights[3]:.4f}")
    print(f"   (Bias/Intercept):  {intercept:.4f}")
    print("="*40 + "\n")
    
    # 5. Save Model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(weights, f)
        
    print(f"Trained weights saved to: {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    if clip_engine is None:
        print("Warning: CLIP Engine not loaded")
    train_ranker()