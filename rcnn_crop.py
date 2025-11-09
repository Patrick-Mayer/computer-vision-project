import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

def get_predictions(image_tensor):
    with torch.no_grad():
        outputs = detector(image_tensor)[0]
    
    threshold = 0.9
    keep = outputs["scores"] >= threshold
    
    return {
        "boxes": outputs["boxes"][keep],
        "masks": outputs["masks"][keep],
        "labels": outputs["labels"][keep],
        "scores": outputs["scores"][keep]
    }

def apply_mask_and_save_refined(original_img, mask_tensor, label_name, index, is_background=False):
    """
    Applies a segmentation mask to the original image, refines the bounding box 
    using the mask pixels, and saves the result.
    """
    
    # Scale and Binarize the mask to original resolution
    mask_pil = transforms.ToPILImage()(mask_tensor[0, :, :]).resize(
        (original_width, original_height), resample=Image.NEAREST
    )
    binary_mask = np.array(mask_pil) > 0 # Boolean mask for the object
    
    if is_background:
        # Invert the mask for the background cutout
        binary_mask = ~binary_mask

    # 2. Convert original image to RGBA format for transparency
    img_rgba = original_img.convert("RGBA")
    img_np = np.array(img_rgba)

    # 3. Apply the mask to the Alpha channel
    img_np[:, :, 3] = binary_mask * 255
    masked_img_pil = Image.fromarray(img_np)

    # 4. Find a TIGHTER Bounding Box based on the *actual mask* pixels using numpy
    # np.where returns coordinates where the condition (mask is True) is met
    coords = np.where(binary_mask)
    if coords[0].size == 0:
        print(f"Skipping empty mask for {label_name}")
        return None

    # Get the tightest possible bounds from the mask coordinates
    ymin, ymax = coords[0].min(), coords[0].max()
    xmin, xmax = coords[1].min(), coords[1].max()
    
    # Add a small padding
    padding = 5
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(original_width, xmax + padding)
    ymax = min(original_height, ymax + padding)

    crop_area_tight = (xmin, ymin, xmax, ymax)
    
    # Perform the crop using PIL's .crop() method on the masked image
    cropped_object = masked_img_pil.crop(crop_area_tight)
    
    # Save the result
    suffix = "background" if is_background else f"{label_name}_{index}"
    save_path = os.path.join(crop_dir, f"masked_crop_{suffix}.png")
    cropped_object.save(save_path)
    print(f"Saved refined segmented crop: {save_path} with tight bounds: {crop_area_tight}")
    
    # Return the boolean mask for use in the main loop's background accumulation
    return binary_mask

def save_background_refined(original_img, accumulated_mask_np):
    # Find tight bounds for the background area
    coords = np.where(~accumulated_mask_np) # Invert mask for background bounds
    if coords[0].size == 0: return

    ymin, ymax = coords[0].min(), coords[0].max()
    xmin, xmax = coords[1].min(), coords[1].max()
    
    padding = 5
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(original_width, xmax + padding)
    ymax = min(original_height, ymax + padding)
    crop_area_tight = (xmin, ymin, xmax, ymax)

    img_rgba = original_img.convert("RGBA")
    img_np_bg = np.array(img_rgba)
    img_np_bg[:, :, 3] = (~accumulated_mask_np) * 255 # Apply inverse mask
    background_pil = Image.fromarray(img_np_bg)
    
    # Crop the background tightly
    cropped_background = background_pil.crop(crop_area_tight)

    save_path = os.path.join(crop_dir, "background.png")
    cropped_background.save(save_path)

# --- Main Execution ---
weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
detector = maskrcnn_resnet50_fpn_v2(weights=weights).eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

crop_dir = "masked_objects_refined"
os.makedirs(crop_dir, exist_ok=True)

img = Image.open("cow.png").convert("RGB")

original_width, original_height = img.size
model_input_size = 256
width_scale = original_width / model_input_size
height_scale = original_height / model_input_size

input_tensor = transform(img).unsqueeze(0)
preds = get_predictions(input_tensor)

all_foreground_masks_np = np.zeros((original_height, original_width), dtype=bool)

for i in range(len(preds["boxes"])):
    label_name = weights.meta['categories'][preds["labels"][i].item()]
        
    # Get and save the refined foreground objects
    current_mask = apply_mask_and_save_refined(
        img, 
        preds["masks"][i], 
        label_name, 
        i+1
    )
    
    if current_mask is not None:
        # Accumulate the total area occupied by all objects
        all_foreground_masks_np = np.logical_or(all_foreground_masks_np, current_mask)

save_background_refined(img, all_foreground_masks_np)