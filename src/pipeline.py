import os
import glob
import sys
import itertools
import joblib
import random
import numpy as np
import pickle

# Adds the src to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import feature extractors from your existing modules
try:
    from src.utils.clip_engine import clip_engine
    from src.utils.feature_extraction import (
        calculate_coherence_score,
        calculate_color_harmony,
        calculate_visual_balance
    )
except ImportError as e:
    print(f"Warning: Module import failed ({e}). Running in limited mode")
    clip_engine = None

class WardrobePipeline:
    def __init__(self, model_path='models/ranker_weights.pkl'):
        """
        Initializes the Pipeline.
        1. Loads the Trained Weights (from train_ranker.py).
        2. Loads the Wardrobe Images from data/test/items.
        """
        print("Initializing Wardrobe Pipeline")
        
        # Load Trained Weights
        # Your train_ranker.py saves a list/array of coefficients, NOT a full model object.
        if os.path.exists(model_path):
            print(f"Loading trained weights from {model_path}")
            with open(model_path, 'rb') as f:
                self.weights = pickle.load(f)
            print(f"Weights loaded: {self.weights}")
        else:
            print(f"Warning: Weights file '{model_path}' not found. Using default equal weights")
            # Default fallback: [Style, Coherence, Color, Balance]
            self.weights = np.array([0.25, 0.25, 0.25, 0.25])

        # Load User's Wardrobe
        self.wardrobe = self._load_wardrobe_database()
        print(f"Loaded {len(self.wardrobe['top'])} tops, {len(self.wardrobe['bottom'])} bottoms, {len(self.wardrobe['shoes'])} shoes")

    def _load_wardrobe_database(self):
        """
        Scans data/test/items/ to build a list of available items.
        Supports .jpg, .jpeg, and .png.
        """
        wardrobe = {'top': [], 'bottom': [], 'shoes': []}
        base_path = "data/test/items"
        
        # Ensure directories exist so we don't crash on empty folders
        for cat in wardrobe.keys():
            os.makedirs(os.path.join(base_path, cat), exist_ok=True)
        
        for category in wardrobe.keys():
            # Scan for multiple image formats
            files = []
            for ext in ["*.jpg", "*.jpeg", "*.png"]:
                files.extend(glob.glob(os.path.join(base_path, category, ext)))
            
            # Mock data generator (if folders are empty)
            if not files: 
                print(f"No images found in {category}, generating mocks for testing")
                for i in range(2): 
                    wardrobe[category].append({
                        'id': f"mock_{category}_{i}",
                        'path': f"mock_path/{category}_{i}.jpg", 
                        'embedding': None
                    })
                continue

            for file_path in files:
                # Use CLIP Engine to get embeddings if available
                emb = None
                if clip_engine:
                    try:
                        emb = clip_engine.get_image_embedding(file_path)
                    except Exception as e:
                        print(f"Warning: Could not get embedding for {file_path}: {e}")

                wardrobe[category].append({
                    'id': os.path.basename(file_path),
                    'path': file_path,
                    'embedding': emb
                })
                
        return wardrobe

    def extract_features_for_items(self, top, bottom, shoe):
        """
        Extracts features using REAL data via src.utils.feature_extraction
        """
        # Prepare inputs for your existing functions
        embeddings = []
        if top['embedding'] is not None: embeddings.append(top['embedding'])
        if bottom['embedding'] is not None: embeddings.append(bottom['embedding'])
        if shoe['embedding'] is not None: embeddings.append(shoe['embedding'])
        
        image_paths = [top['path'], bottom['path'], shoe['path']]
        
        # STYLE FIT
        # Since we don't have a specific user style vector yet, we simulate or assume neutral
        style_fit = 0.8 # Assume good fit for now
        
        # COHERENCE
        if len(embeddings) == 3:
            coherence = calculate_coherence_score(embeddings)
        else:
            # Fallback if CLIP failed
            coherence = 0.5 
            
        # COLOR HARMONY
        try:
            color = calculate_color_harmony(image_paths)
        except Exception as e:
            print(f"Color calc error: {e}")
            color = 0.5
            
        # VISUAL BALANCE
        try:
            balance = calculate_visual_balance(image_paths)
        except Exception as e:
            print(f"Balance calc error: {e}")
            balance = 0.5
        
        return np.array([style_fit, coherence, color, balance])

    def calculate_outfit_similarity(self, outfit_a, outfit_b):
        """
        Calculates cosine similarity between two outfits based on their CLIP embeddings.
        This is used for MMR to determine how 'visually similar' two results are.
        """
        # Collect valid embeddings for each outfit
        # outfit['items'] is the list [top, bottom, shoe]
        vecs_a = []
        vecs_b = []
        
        for item_a, item_b in zip(outfit_a['items'], outfit_b['items']):
            if item_a['embedding'] is not None and item_b['embedding'] is not None:
                vecs_a.append(item_a['embedding'])
                vecs_b.append(item_b['embedding'])
        
        # If no embeddings (e.g. mock data or CLIP failed), we assume they are different
        if not vecs_a or not vecs_b:
            return 0.0
            
        # Strategy: Concatenate embeddings to form a single "Outfit Vector" (e.g. 512*3 = 1536 dim)
        flat_a = np.concatenate(vecs_a)
        flat_b = np.concatenate(vecs_b)
        
        norm_a = np.linalg.norm(flat_a)
        norm_b = np.linalg.norm(flat_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return np.dot(flat_a, flat_b) / (norm_a * norm_b)

    def apply_mmr(self, ranked_candidates, top_k=5, lambda_param=0.34):
        """
        Maximal Marginal Relevance (MMR) Selection.
        
        Args:
            ranked_candidates: List of outfit dicts (must have 'score' and 'items' with embeddings).
            top_k: How many outfits to return.
            lambda_param: Trade-off parameter (0.0 to 1.0).
                          1.0 = Pure Score (Standard Ranking)
                          0.5 = Balance between Score and Diversity
                          0.0 = Pure Diversity (Novelty)
        """
        if not ranked_candidates:
            return []
            
        selected = []
        pool = ranked_candidates.copy()
        
        # We assume the input is already sorted by score, but it doesn't strictly matter for MMR
        
        while len(selected) < top_k and pool:
            best_mmr_score = -float('inf')
            best_candidate = None
            
            for candidate in pool:
                # 1. Relevance (The model's predicted score)
                # Assumes score is roughly 0.0 to 1.0
                relevance = candidate['score']
                
                # 2. Diversity Penalty (Max similarity to any already selected item)
                if not selected:
                    diversity_penalty = 0.0
                else:
                    similarities = [self.calculate_outfit_similarity(candidate, s) for s in selected]
                    diversity_penalty = max(similarities)
                
                # MMR Equation
                mmr_score = (lambda_param * relevance) - ((1 - lambda_param) * diversity_penalty)
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate
            
            # Add the best candidate to selected list
            if best_candidate:
                selected.append(best_candidate)
                pool.remove(best_candidate)
                
        return selected

    def run_ranking(self, top_k=5, diversity_lambda=0.6):
        """
        1. Generates combinations.
        2. Extracts features.
        3. Ranks using the trained weights.
        4. Applies MMR for diversity.
        """
        print(f"\n⚡ Running Wardrobe Ranking...")
        
        tops = self.wardrobe['top']
        bottoms = self.wardrobe['bottom']
        shoes = self.wardrobe['shoes']
        
        if not tops or not bottoms or not shoes:
            print("Error: Wardrobe is incomplete (missing categories).")
            return []

        ranked_outfits = []
        
        # Generate Combinations
        combinations = list(itertools.product(tops, bottoms, shoes))
        print(f"Analyzing {len(combinations)} outfits...")
        
        for outfit in combinations:
            top, bottom, shoe = outfit
            
            # Extract Features (The Real Pipeline Step)
            features = self.extract_features_for_items(top, bottom, shoe)
            
            # Score with dot product
            # Handling potential shape mismatch if bias was saved or not
            if len(self.weights) == 4:
                score = np.dot(self.weights, features)
            elif len(self.weights) == 5:
                # Assuming first 4 are weights, last is intercept
                score = np.dot(self.weights[:4], features)
            else:
                score = np.mean(features) # Fallback

            ranked_outfits.append({
                'items': [top, bottom, shoe],
                'score': score,
                'features': features
            })
        
        # Sort by Score (Highest first) - Creates the initial pool
        ranked_outfits.sort(key=lambda x: x['score'], reverse=True)
        
        # Apply MMR for Diversity
        print(f"Applying MMR Re-ranking (lambda={diversity_lambda})...")
        final_selection = self.apply_mmr(ranked_outfits, top_k=top_k, lambda_param=diversity_lambda)
        
        return final_selection


if __name__ == "__main__":
    try:
        # Start the pipeline
        pipeline = WardrobePipeline(model_path='models/ranker_weights.pkl')
        
        # Run ranking with MMR
        # lambda=0.6 leans slightly towards quality but ensures we don't pick duplicates
        results = pipeline.run_ranking(top_k=5, diversity_lambda=0.6)
        
        # Print Results
        print(f"\nTop {len(results)} Diverse & Compatible Outfits:")
        for i, outfit in enumerate(results):
            items_ids = [item['id'] for item in outfit['items']]
            print(f"\n#{i+1}: {' + '.join(items_ids)}")
            print(f"    Score: {outfit['score']:.4f}")
            f = outfit['features']
            print(f"    [Features] Style:{f[0]:.2f}, Coh:{f[1]:.2f}, Color:{f[2]:.2f}, Bal:{f[3]:.2f}")
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()