import torch
import clip
import numpy as np
from PIL import Image


class CLIPEngine:
    def __init__(self, model_name="ViT-B/32"):
        """
        Initializes the CLIP model using the official OpenAI repository.
        
        Args:
            model_name (str): The model architecture to use. 
                              Options: 'ViT-B/32', 'ViT-L/14', 'RN50', etc.
        """
        print(f"Loading OpenAI CLIP model: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.model.eval() # Set to evaluation mode
            print(f"CLIP Loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading CLIP: {e}")
            self.model = None

    def get_image_embedding(self, image_source):
        """
        Extracts 512D embedding from an image.
        
        Args:
            image_source: File path (str) OR PIL Image object
        Returns:
            Normalized Numpy array of shape (512,)
        """
        if self.model is None:
            return np.zeros(512)

        # Load Image
        image = None
        if isinstance(image_source, str):
            try:
                image = Image.open(image_source).convert("RGB")
            except Exception as e:
                print(f"Error loading image {image_source}: {e}")
                return np.zeros(512)
        elif isinstance(image_source, Image.Image):
            image = image_source.convert("RGB")
        else:
            print("Invalid image source format")
            return np.zeros(512)

        # Preprocess & Forward Pass
        try:
            # Preprocess returns a tensor (3, H, W). We need to unsqueeze to make it a batch (1, 3, H, W).
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)

            # Normalize vector (Critical for Cosine Similarity)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            # Convert to flattened numpy array
            return image_features.cpu().numpy()[0]
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return np.zeros(512)

    def get_text_embedding(self, text):
        """
        Extracts embedding from text (e.g., "formal style", "streetwear").
        """
        if self.model is None:
            return np.zeros(512)

        try:
            # Tokenize text (truncates if too long automatically)
            text_input = clip.tokenize([text]).to(self.device)

            with torch.no_grad():
                text_features = self.model.encode_text(text_input)

            # Normalize
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy()[0]
            
        except Exception as e:
            print(f"Error extracting text embedding: {e}")
            return np.zeros(512)


# Singleton instance
try:
    clip_engine = CLIPEngine()
except Exception:
    clip_engine = None