import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine


"""
FEATURE EXTRACTION MODULE FOR OUTFIT RANKING

This module calculates the 3 key "human-like" metrics for an outfit:
1. Coherence (Style Consistency) - using CLIP vectors
2. Color Harmony (Color Theory) - using OpenCV & HSV
3. Visual Balance (Complexity) - using Edge Density
"""


# COHERENCE
def calculate_coherence_score(embeddings):
    """
    Calculates how "consistent" the vibe is across items.
    Input: List of CLIP embeddings (numpy arrays) for [top, bottom, shoes]
    Output: Score 0.0 to 1.0
    """
    if len(embeddings) < 2:
        return 1.0
    
    similarities = []
    # Calculate pairwise cosine similarity between all items
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            # 1 - cosine_distance = cosine_similarity
            sim = 1 - cosine(embeddings[i], embeddings[j])
            similarities.append(sim)
            
    # Average similarity
    return np.mean(similarities)

# COLOR HARMONY
def get_dominant_color(image_path, k=1):
    """Extracts dominant color in HSV format"""
    # Load image
    img = cv2.imread(image_path)
    if img is None: return (0, 0, 0)
    
    # Convert BGR to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Reshape to a list of pixels
    pixels = img.reshape((-1, 3))
    
    # Use K-Means to find the most common color
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(pixels)
    
    # Dominant color (Hue, Saturation, Value)
    dominant_color = kmeans.cluster_centers_[0]
    return dominant_color

def calculate_color_harmony(image_paths):
    """
    Scores outfit based on color theory rules (Analogous/Complementary)
    Input: List of file paths to item images
    Output: Score 0.0 to 1.0
    """
    colors = [get_dominant_color(p) for p in image_paths]
    
    if len(colors) < 2: return 1.0

    # Extract Hues (0-179)
    hues = [c[0] for c in colors]
    saturations = [c[1] for c in colors]
    
    # Calculate Hue Difference between Top and Bottom
    # Normalize to 360-degree circle (0-179*2)
    hue_top = hues[0] * 2
    hue_bottom = hues[1] * 2
    
    diff = abs(hue_top - hue_bottom)
    if diff > 180: diff = 360 - diff
    
    # Color theory rules
    score = 0.5 # Default neutral
    
    # Monochromatic / Analogous (Similar colors) -> Safe & High Score
    if diff < 30:
        score = 1.0
        
    # Complementary (Opposite colors, e.g., Blue & Orange) -> High Score
    elif 150 < diff < 210:
        score = 0.9
        
    # Triadic/Square (90 degree clashes) -> Often tricky/low score
    elif 80 < diff < 100:
        score = 0.4
        
    # Penalty for low saturation (grey/muddy colors matching poorly)
    if np.mean(saturations) < 20: 
        score = 0.8 # Greyscale matching is usually fine
        
    return score


# VISUAL BALANCE
def get_edge_density(image_path):
    """Calculates 'busyness' of an item using Canny Edge Detection"""
    img = cv2.imread(image_path, 0) # Read as grayscale
    if img is None: return 0
    
    # Blur to remove noise
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    # Canny Edge Detection
    edges = cv2.Canny(blurred, 100, 200)
    
    # Calculate density: ratio of white pixels (edges) to total pixels
    density = np.sum(edges) / (edges.size * 255)
    return density

def calculate_visual_balance(image_paths):
    """
    Ensures the outfit isn't too 'busy'.
    Rule: Avoid combining multiple high-pattern items.
    """
    densities = [get_edge_density(p) for p in image_paths]
    
    # Threshold for "Busy" item (patterned/complex)
    BUSY_THRESHOLD = 0.15 
    
    busy_items_count = sum(1 for d in densities if d > BUSY_THRESHOLD)
    
    # If more than 1 item is very busy (e.g., Plaid Shirt + Floral Pants) -> Penalty
    if busy_items_count > 1:
        return 0.3 # Low score for clashing patterns
    
    # If 1 busy item + others simple -> High score (Balanced)
    elif busy_items_count == 1:
        return 1.0
        
    # If all simple (Minimalist) -> Good score
    return 0.9