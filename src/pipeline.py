import numpy as np
import os
import glob
import sys

# Adds the src to PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our custom modules
from src.utils.clip_engine import clip_engine
from src.ml_modules.model_4_ranking import OutfitRanker

class Model4TestPipeline:
    def __init__(self):
        """
        Initializes the Test Pipeline for Model 4.
        Focuses purely on Wardrobe Loading + Ranking.
        """
        print("Initializing Model 4 Test Pipeline")
        
        # Ensure CLIP is loaded (Needed for Coherence check inside Ranker)
        if clip_engine is None:
            raise Exception("CRITICAL: CLIP Engine failed to load.")
            
        # Initialize the Ranker (Model 4)
        self.ranker = OutfitRanker()
        
        # Load User's Wardrobe
        self.wardrobe = self._load_wardrobe_database()
        print(f"Loaded {len(self.wardrobe['top'])} tops, {len(self.wardrobe['bottom'])} bottoms, {len(self.wardrobe['shoes'])} shoes.")

    def _load_wardrobe_database(self):
        """
        Scans data/user/items/ to build a list of available items.
        """
        wardrobe = {'top': [], 'bottom': [], 'shoes': []}
        base_path = "data/user/items"
        
        for category in wardrobe.keys():
            path = os.path.join(base_path, category, "*.jpg")
            files = glob.glob(path)
            
            # Test data generator
            if not files: 
                for i in range(3): # Create 3 mock items per category
                    wardrobe[category].append({
                        'id': f"mock_{category}_{i}",
                        'path': f"mock_path/{category}_{i}.jpg", # Fake path
                        'embedding': np.random.rand(512) # Fake embedding
                    })
                continue

            for file_path in files:
                # Perform embedding on the items
                emb = clip_engine.get_image_embedding(file_path)
                wardrobe[category].append({
                    'id': os.path.basename(file_path),
                    'path': file_path,
                    'embedding': emb
                })
                
        return wardrobe

    def run_compatibility_test(self):
        """
        Execute Model 4
        1. Takes all user items.
        2. Generates combinations.
        3. Ranks them by Coherence, Color, and Balance.
        """
        print(f"\n⚡ Running Compatibility Test (Model 4 Only)...")
        
        # Get All Items (No Filtering by Style)
        tops = self.wardrobe['top']
        bottoms = self.wardrobe['bottom']
        shoes = self.wardrobe['shoes']
        
        if not tops or not bottoms or not shoes:
            print("Error: Wardrobe is empty. Please add images to 'data/user/items/...'")
            return []

        # Rank Outfits
        ranked_outfits = self.ranker.rank_outfits(
            tops, 
            bottoms, 
            shoes, 
            target_style_emb=None, # Currently no external style target (not trying to match KOL)
            top_k=10
        )
        
        return ranked_outfits


if __name__ == "__main__":
    # Start the test pipeline
    pipeline = Model4TestPipeline()
    
    # Run the ranker on the user's items
    results = pipeline.run_compatibility_test()
    
    # Print Results
    print(f"\nTop {len(results)} Compatible Outfits from User's Closet:")
    for i, outfit in enumerate(results):
        print(f"\n#{i+1}: Total Score: {outfit['score']:.4f}")
        print(f"    Items: {outfit['items'][0]['id']} + {outfit['items'][1]['id']} + {outfit['items'][2]['id']}")
        print("    [Analysis]:")
        print(f"      - Coherence (Vibe match):   {outfit['details']['coherence']:.2f}") 
        print(f"      - Color Harmony (HSV):      {outfit['details']['color']:.2f}")
        print(f"      - Visual Balance (Clutter): {outfit['details']['balance']:.2f}")