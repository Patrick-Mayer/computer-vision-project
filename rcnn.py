import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as TF

# Pretrained weights and R-CNN model from PyTorch
weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = maskrcnn_resnet50_fpn_v2(weights=weights).eval()

model.eval()

# img = Image.open("chihuahua.png").convert("RGB")
img = Image.open("cow.png").convert("RGB")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(), # Keeps image from getting distorted
])
input_tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)[0]

# Extract predictions
masks = outputs["masks"]
labels = outputs["labels"]
scores = outputs["scores"]

# Keep only high confidence detections
threshold = 0.88 # Change this if objects aren't being detected
keep = scores >= threshold

masks = masks[keep]
labels = labels[keep]
scores = scores[keep]

# Combine masks into single map
if len(masks) > 0:
    combined_mask = torch.zeros_like(masks[0, 0])
    for i, mask in enumerate(masks):
        combined_mask += (mask[0] > 0.7).float() * (i + 1)
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

# Draw bounding boxes and labels
# Convert image to tensor (0-255, uint8)
img_tensor = (input_tensor[0] * 255).byte()

labels_str = [f"{weights.meta['categories'][lbl]}: {score:.2f}" for lbl, score in zip(labels, scores)]
boxed = draw_bounding_boxes(img_tensor, outputs["boxes"][keep], labels=labels_str, colors="red", width=2)
plt.imshow(TF.to_pil_image(boxed))
plt.axis("off")
plt.title("Detected Objects")
plt.show()