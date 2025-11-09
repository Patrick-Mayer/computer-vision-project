import clip as cl
import torch
from PIL import Image

# Load model ViT-L/14 for best similarity matching results
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = cl.load("ViT-L/14", device=device)

# Load and preprocess images
img1 = preprocess(Image.open("chihuahua.png")).unsqueeze(0).to(device)
img2 = preprocess(Image.open("cow.png")).unsqueeze(0).to(device)

# Get embeddings
with torch.no_grad():
    emb1 = model.encode_image(img1)
    emb2 = model.encode_image(img2)

# Normalize and compare
emb1 /= emb1.norm(dim=-1, keepdim=True)
emb2 /= emb2.norm(dim=-1, keepdim=True)
# Cosine similarity Ab / ||A||||B||
similarity = (emb1 @ emb2.T).item()

print(f"Similarity: {similarity:.2f}")



