import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as TF

def get_mask(img):
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)[0]

    # Extract predictions
    boxes = outputs["boxes"]
    masks = outputs["masks"]
    labels = outputs["labels"]
    scores = outputs["scores"]

    # Keep only high confidence detections
    threshold = 0.88 # Change this if objects aren't being detected
    keep = scores >= threshold

    boxes = boxes[keep]
    masks = masks[keep]
    labels = labels[keep]
    scores = scores[keep]

    # We need to scale the box coordinates from the 256x256 model input size 
    # back to the original image's resolution.
    original_width, original_height = img.size
    model_input_size = 256

    width_scale = original_width / model_input_size
    height_scale = original_height / model_input_size

    # Extract bounding box coords
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box.tolist()

        # Scale coordinates back to the original image dimensions
        xmin_scaled = int(xmin * width_scale)
        ymin_scaled = int(ymin * height_scale)
        xmax_scaled = int(xmax * width_scale)
        ymax_scaled = int(ymax * height_scale)

        # PIL uses a tuple of (left, upper, right, lower)
        crop_area = (xmin_scaled, ymin_scaled, xmax_scaled, ymax_scaled)
        
        # Perform the crop using PIL's .crop() method on the ORIGINAL image
        cropped_img = img.crop(crop_area)

        label_name = weights.meta['categories'][labels[i].item()]
        score_val = scores[i].item()
        cropped_img.save(f"cropped_{label_name}.png")
        print(f"Object: {label_name}, Score: {score_val:.2f}")
        print(f"Box Coordinates: (x_min={xmin:.2f}, y_min={ymin:.2f}), (x_max={xmax:.2f}, y_max={ymax:.2f})\n")

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
    boxed = draw_bounding_boxes(img_tensor, boxes, labels=labels_str, colors="red", width=2)
    plt.imshow(TF.to_pil_image(boxed))
    plt.axis("off")
    plt.title("Detected Objects")
    plt.show()

# Pretrained weights and R-CNN model from PyTorch
weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = maskrcnn_resnet50_fpn_v2(weights=weights).eval()

# img = Image.open("chihuahua.png").convert("RGB")
img = Image.open("chihuahua.png").convert("RGB")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(), # Keeps image from getting distorted
])

get_mask(img)