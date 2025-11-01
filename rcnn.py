import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

# Pretrained weights and R-CNN model from PyTorch
weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
model = maskrcnn_resnet50_fpn(weights=weights).eval()

model.eval()

img = Image.open("chihuahua.png").convert("RGB")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)[0]

# Extract predictions
masks = outputs["masks"]
labels = outputs["labels"]
scores = outputs["scores"]

# Keep only high confidence detections
threshold = 0.8
keep = scores >= threshold

masks = masks[keep]
labels = labels[keep]
scores = scores[keep]

# Combine masks into single map
if len(masks) > 0:
    combined_mask = torch.zeros_like(masks[0, 0])
    for i, mask in enumerate(masks):
        combined_mask += (mask[0] > 0.5).float() * (i + 1)
    pred_mask = combined_mask.cpu().numpy()
else:
    pred_mask = torch.zeros((256, 256)).numpy()

# Plot images
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Predicted Mask (Segmented)")
plt.imshow(pred_mask)
plt.show()
