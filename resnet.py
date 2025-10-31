import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights # We can look at other models

# Pretrained weights and Res-Net model from PyTorch
# Res-Net can be used to build a U-Net
weights=DeepLabV3_ResNet50_Weights.DEFAULT
model = deeplabv3_resnet50(pretrained=True, weights=weights)

model.eval()

img = Image.open("chihuahua.png").convert("RGB")

input_tensor = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)["out"]
    pred_mask = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

# Plot images
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Pedicted Mask (Segmented)")
plt.imshow(pred_mask)
plt.show()