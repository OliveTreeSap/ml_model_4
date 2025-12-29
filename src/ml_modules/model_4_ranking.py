import pickle
import itertools
import numpy as np
import os
import sys

# Adds the src to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the feature extractors
from src.utils.feature_extraction import (
    calculate_coherence_score,
    calculate_color_harmony,
    calculate_visual_balance
)

# Constants
WEIGHTS_PATH = "models/ranker_weights.pkl"

class OutfitRanker:
    def __init__(self, weights_path=WEIGHTS_PATH):
        self.weights = self._load_weights(weights_path)
        print(f"OutfitRanker loaded with weights: {self.weights}")

    def _load_weights(self, path):
        """
        Loads the learned weights from the offline training phase.
        If no file exists, defaults to a heuristic.
        """
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        else:
            print("Warning: Trained weights not found. Using default heuristic.")
            # Default: [Style, Coherence, Color, Balance]
            return np.array([0.2, 0.4, 0.3, 0.1]) 

    def predict_compatibility_score(self, top, bottom, shoes, target_style_emb=None):
        """
        Calculates the score for a single outfit combination.
        Focuses on how well items mesh (Coherence, Color, Balance).
        """
        # Prepare Inputs
        embeddings = [top['embedding'], bottom['embedding'], shoes['embedding']]
        image_paths = [top['path'], bottom['path'], shoes['path']]

        # Extract Features
        
        # Coherence
        coherence = calculate_coherence_score(embeddings)
        
        # Color Harmony
        color_harmony = calculate_color_harmony(image_paths)
        
        # Visual Balance
        balance = calculate_visual_balance(image_paths)

        # Style Fit (Optional depending on the output of previous models)
        style_fit = 0.0
        if target_style_emb is not None:
            # Average cosine similarity to target
            sims = [np.dot(e, target_style_emb) for e in embeddings]
            style_fit = np.mean(sims)

        # Calculate Final Weighted Score
        # Features vector: [Style, Coherence, Color, Balance]
        features = np.array([style_fit, coherence, color_harmony, balance])
        
        # Dot product
        final_score = np.dot(self.weights, features)
        
        return final_score, features

    def rank_outfits(self, candidate_tops, candidate_bottoms, candidate_shoes, target_style_emb=None, top_k=5):
        """
        Returns the top K best outfits by:
        1. Generates all combinations (Cartesian Product).
        2. Scores and sort them based on compatibility.
        """
        scored_outfits = []

        # Generate Combinations
        combinations = list(itertools.product(candidate_tops, candidate_bottoms, candidate_shoes))
        
        print(f"Analyzing {len(combinations)} possible combinations")

        for top, bottom, shoes in combinations:
            score, debug_features = self.predict_compatibility_score(
                top, bottom, shoes, target_style_emb
            )
            
            scored_outfits.append({
                'items': [top, bottom, shoes],
                'score': score,
                'details': {
                    'coherence': debug_features[1],
                    'color': debug_features[2],
                    'balance': debug_features[3]
                }
            })

        # Sort by Score (Descending)
        scored_outfits.sort(key=lambda x: x['score'], reverse=True)

        # Return Top K
        return scored_outfits[:top_k]


if __name__ == "__main__":
    mock_tops = [{'id': 't1', 'path': 'data/user/items/top/t1.jpg', 'embedding': np.random.rand(512)}]
    mock_bottoms = [{'id': 'b1', 'path': 'data/user/items/bottom/b1.jpg', 'embedding': np.random.rand(512)}]
    mock_shoes = [{'id': 's1', 'path': 'data/user/items/shoes/s1.jpg', 'embedding': np.random.rand(512)}]

    ranker = OutfitRanker()
    best_outfits = ranker.rank_outfits(mock_tops, mock_bottoms, mock_shoes)
    
    print(f"Top Outfit Score: {best_outfits[0]['score']:.4f}")