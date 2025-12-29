import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression

# CONFIGURATION
MODEL_SAVE_PATH = "models/ranker_weights.pkl"
N_SAMPLES = 1000  # Number of random outfits to synthesize for training

def generate_synthetic_outfit_features(n_samples):
    """
    Step 1: DATA GENERATION
    Simulates the features of random outfits created from the closet.
    
    In a real production run with 10,000 images, you would:
    1. Load real images.
    2. Run feature_extraction.py on them.
    3. Use those results here.
    
    For this project (Demo), we simulate the feature distribution using 
    normal distributions to represent realistic wardrobe combinations.
    """
    print(f"🔄 Synthesizing {n_samples} outfit combinations...")
    
    # Feature 1: Style Fit (0.0 - 1.0)
    # Most random combos won't match the target style perfectly, so mean is 0.5
    style_fits = np.random.normal(0.5, 0.25, n_samples)
    
    # Feature 2: Coherence (0.0 - 1.0)
    # Random items usually don't vibe well, so mean is slightly lower
    coherences = np.random.normal(0.45, 0.2, n_samples) 
    
    # Feature 3: Color Harmony (0.0 - 1.0)
    # Random colors often clash, but sometimes match by luck
    colors = np.random.normal(0.5, 0.3, n_samples)
    
    # Feature 4: Visual Balance (0.0 - 1.0)
    # Most items are not super complex, so balance is usually decent
    balances = np.random.normal(0.6, 0.2, n_samples)
    
    # Stack into a matrix (N, 4) and clip to valid range [0, 1]
    X_features = np.column_stack([style_fits, coherences, colors, balances])
    X_features = np.clip(X_features, 0.0, 1.0)
    
    return X_features

def expert_labeling_teacher(features):
    """
    Step 2: PSEUDO-LABELING (The Self-Supervised 'Teacher')
    
    This function represents the 'Expert Knowledge' injected into the system.
    It automatically decides if an outfit is 'Good' or 'Bad' based on strict rules.
    """
    style, coh, color, bal = features
    
    # --- RULE SET ---
    
    # Rule A: The "Perfect Outfit" (Label 1)
    # Must be stylish AND coherent AND matching colors
    if style > 0.75 and coh > 0.65 and color > 0.6:
        return 1
    
    # Rule B: The "Disaster Outfit" (Label 0)
    # Fails if it misses the style badly OR colors clash horribly OR it's too messy
    elif style < 0.45 or color < 0.35 or bal < 0.3:
        return 0
        
    # Rule C: Ambiguous Case (Label -1)
    # If it's "okay but not great", we skip it to avoid confusing the model
    else:
        return -1

def train_ranker():
    # 1. Generate Data
    X_raw = generate_synthetic_outfit_features(N_SAMPLES)
    
    X_train = []
    y_train = []
    
    # 2. Apply Self-Supervision
    print("🏷️  Teacher is labeling the random outfits...")
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
    
    print(f"   ✅ Data labeled. Good: {count_good}, Bad: {count_bad}, Total: {len(X_train)}")

    # 3. Train the Student Model (Logistic Regression)
    # We set fit_intercept=False to force the model to learn raw feature weights
    print("🧠 Training Student Model (Logistic Regression)...")
    clf = LogisticRegression(fit_intercept=False, solver='lbfgs')
    clf.fit(X_train, y_train)
    
    # 4. Extract and Display Learned Weights
    # These w values represent how important the model thinks each feature is
    weights = clf.coef_[0]
    
    print("\n" + "="*40)
    print("🚀 SELF-SUPERVISED TRAINING COMPLETE")
    print("   The model has learned that a good outfit depends on:")
    print(f"   1. Style Fit:      {weights[0]:.4f}")
    print(f"   2. Coherence:      {weights[1]:.4f}")
    print(f"   3. Color Harmony:  {weights[2]:.4f}")
    print(f"   4. Visual Balance: {weights[3]:.4f}")
    print("="*40 + "\n")
    
    # 5. Save the 'Brain' (Weights) to disk
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(weights, f)
        
    print(f"💾 Trained weights saved to: {MODEL_SAVE_PATH}")
    print("   (You can now run 'src/ml_modules/idea_4_ranking.py' to use these weights)")

if __name__ == "__main__":
    train_ranker()