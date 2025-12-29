import os
import glob
import sys
import itertools
import joblib
import random
import numpy as np


# Adds the src to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.utils.clip_engine import clip_engine
except ImportError:
    clip_engine = None
    print("Warning: CLIP Engine not found. Running in simulation mode if needed.")

class WardrobePipeline:
    def __init__(self, model_path='wardrobe_ranker.pkl'):
        """
        Initializes the Pipeline.
        1. Loads the Trained Model in models.
        2. Loads the Wardrobe Images from data/user/items.
        """
        print("Initializing Wardrobe Pipeline")
        
        # Load Trained Model
        if os.path.exists(model_path):
            print(f"Loading trained model from {model_path}")
            self.model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file '{model_path}' not found. Please run 'train_ranker.py' first to generate it")

        # Load User's Wardrobe
        self.wardrobe = self._load_wardrobe_database()
        print(f"Loaded {len(self.wardrobe['top'])} tops, {len(self.wardrobe['bottom'])} bottoms, {len(self.wardrobe['shoes'])} shoes")

    def _load_wardrobe_database(self):
        """
        Scans data/test/items/ to build a list of available items.
        """
        wardrobe = {'top': [], 'bottom': [], 'shoes': []}
        base_path = "data/test/items"
        
        # Ensure directories exist so we don't crash on empty folders
        for cat in wardrobe.keys():
            os.makedirs(os.path.join(base_path, cat), exist_ok=True)
        
        for category in wardrobe.keys():
            path = os.path.join(base_path, category, "*.jpg")
            files = glob.glob(path)
            
            # Mock data generator (if folders are empty)
            if not files: 
                print(f"No images found in {category}, generating mocks for testing")
                for i in range(2): 
                    wardrobe[category].append({
                        'id': f"mock_{category}_{i}",
                        'path': f"mock_path/{category}_{i}.jpg", 
                    })
                continue

            for file_path in files:
                # If you need embeddings later, you can generate them here:
                # emb = clip_engine.get_image_embedding(file_path) if clip_engine else None
                wardrobe[category].append({
                    'id': os.path.basename(file_path),
                    'path': file_path
                })
                
        return wardrobe

    def extract_features_for_items(self, top, bottom, shoe):
        """
        PLACEHOLDER: This is where your real Computer Vision / Metadata logic goes.
        In a real pipeline, this would load images and run them through your feature extractors.
        """
        # TODO: Connect this to your actual image analysis code using clip_engine.
        # For now, we return random values so the pipeline runs for testing.
        
        # Simulating values for Style, Coherence, Color, Balance
        s = random.uniform(0.3, 0.9)
        c = random.uniform(0.3, 0.9)
        col = random.uniform(0.3, 0.9)
        b = random.uniform(0.3, 0.9)
        return [s, c, col, b]

    def run_ranking(self):
        """
        1. Generates combinations.
        2. Extracts features.
        3. Ranks using the trained model.
        """
        print(f"\n⚡ Running Wardrobe Ranking...")
        
        tops = self.wardrobe['top']
        bottoms = self.wardrobe['bottom']
        shoes = self.wardrobe['shoes']
        
        if not tops or not bottoms or not shoes:
            print("Error: Wardrobe is incomplete (missing categories).")
            return []

        ranked_outfits = []
        
        # 1. Generate Combinations
        combinations = list(itertools.product(tops, bottoms, shoes))
        
        for outfit in combinations:
            top, bottom, shoe = outfit
            
            # 2. Extract Features (The Real Pipeline Step)
            features = self.extract_features_for_items(top, bottom, shoe)
            
            # 3. Score with Model
            # Reshape for model: [[f1, f2, f3, f4]]
            model_input = [features]
            
            try:
                # Get probability of class 1 (Good Outfit)
                score = self.model.predict_proba(model_input)[0][1]
            except AttributeError:
                # Fallback if model doesn't support probability
                score = float(self.model.predict(model_input)[0])
            
            ranked_outfits.append({
                'items': [top, bottom, shoe],
                'score': score,
                'features': features
            })
        
        # 4. Sort by Score (Highest first)
        ranked_outfits.sort(key=lambda x: x['score'], reverse=True)
        
        return ranked_outfits


if __name__ == "__main__":
    try:
        # Start the pipeline
        pipeline = WardrobePipeline()
        
        # Run ranking
        results = pipeline.run_ranking()
        
        # Print Results
        print(f"\nTop {min(5, len(results))} Compatible Outfits:")
        for i, outfit in enumerate(results[:5]):
            items_ids = [item['id'] for item in outfit['items']]
            print(f"\n#{i+1}: {' + '.join(items_ids)}")
            print(f"    Score: {outfit['score']:.4f}")
            f = outfit['features']
            print(f"    [Features] Style:{f[0]:.2f}, Coh:{f[1]:.2f}, Color:{f[2]:.2f}, Bal:{f[3]:.2f}")
            
    except Exception as e:
        print(f"\nAn error occurred: {e}")