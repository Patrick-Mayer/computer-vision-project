import clip
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

def img_tensor(img, og_width, og_height):
    # Values are resizing

    model_input_size = 256
    width_scale = og_width / model_input_size
    height_scale = og_height / model_input_size

    input_tensor = transform(img).unsqueeze(0)

    return input_tensor

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

def apply_mask_and_save_refined(og_img, mask_tensor, label_name, index, is_background, crop_dir, og_width, og_height):
    mask_np = mask_tensor[0].cpu().numpy()
    mask_resized = np.array(Image.fromarray(mask_np).resize((og_width, og_height), resample=Image.BILINEAR))
    binary_mask = mask_resized > 0.5
    
    if is_background:
        # Invert the mask for the background cutout
        binary_mask = ~binary_mask

    # Convert og image to RGBA format for transparency
    img_rgba = og_img.convert("RGBA")
    img_np = np.array(img_rgba)

    # Apply the mask to the Alpha channel
    img_np[:, :, 3] = binary_mask * 255
    masked_img_pil = Image.fromarray(img_np)

    # Find object using bounding box + mask
    coords = np.where(binary_mask) # coords where mask == TRUE
    if coords[0].size == 0:
        print(f"Skipping empty mask for {label_name}")
        return None

    ymin, ymax = coords[0].min(), coords[0].max()
    xmin, xmax = coords[1].min(), coords[1].max()
    
    # Some padding
    padding = 5
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(og_width, xmax + padding)
    ymax = min(og_height, ymax + padding)

    crop_area_tight = (xmin, ymin, xmax, ymax)
    
    # Crop!
    cropped_object = masked_img_pil.crop(crop_area_tight)
    
    # Results
    if is_background:
        suffix = "background"
    else: 
        suffix = f"{label_name}_{index}"

    save_path = os.path.join(crop_dir, f"cropped_mask_{suffix}.png")
    cropped_object.save(save_path)
    
    return binary_mask

def get_objs(og_width, og_height, preds, img, crop_dir): 
    all_foreground_masks_np = np.zeros((og_height, og_width), dtype=bool) 
    objects = [] 

    for i in range(len(preds["boxes"])): 
        label_name = weights.meta['categories'][preds["labels"][i].item()] # Get and save the refined foreground objects 

        obj = apply_mask_and_save_refined( 
            img, 
            preds["masks"][i], 
            label_name, 
            i+1, 
            False, 
            crop_dir, 
            og_width, 
            og_height ) 
        
        if obj is not None: 
            # Total area of objects 
            objects.append(obj) 
            all_foreground_masks_np = np.logical_or(all_foreground_masks_np, obj) 
            
    save_background(img, all_foreground_masks_np, og_width, og_height, crop_dir) 
    return objects

def save_background(og_img, accumulated_mask_np, og_width, og_height, crop_dir):
    # Find tight bounds for the background area
    coords = np.where(~accumulated_mask_np) # Invert mask for background bounds
    if coords[0].size == 0: return

    ymin, ymax = coords[0].min(), coords[0].max()
    xmin, xmax = coords[1].min(), coords[1].max()
    
    # Padding!
    padding = 5
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(og_width, xmax + padding)
    ymax = min(og_height, ymax + padding)
    crop_area_tight = (xmin, ymin, xmax, ymax)

    img_rgba = og_img.convert("RGBA")
    img_np_bg = np.array(img_rgba)
    img_np_bg[:, :, 3] = (~accumulated_mask_np) * 255 # Apply inverse mask
    background_pil = Image.fromarray(img_np_bg)
    
    # Crop the background tightly
    cropped_background = background_pil.crop(crop_area_tight)

    save_path = os.path.join(crop_dir, "background.png")
    cropped_background.save(save_path)

def get_clip_embeddings(crops):
    """Encode object crops using CLIP."""
    tensors = [preprocess(c).unsqueeze(0).to(device) for c in crops]
    imgs = torch.cat(tensors)
    with torch.no_grad():
        feats = clip_model.encode_image(imgs)
    feats /= feats.norm(dim=-1, keepdim=True)
    return feats

weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
detector = maskrcnn_resnet50_fpn_v2(weights=weights).eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)

# Update test image
img1_name = "chihuahua.png"
img2_name = "cow.png"

img1 = Image.open(img1_name).convert("RGB")

crop_dir1 = f"masked_objects_for_{img1_name}"
os.makedirs(crop_dir1, exist_ok=True)

img2 = Image.open(img2_name).convert("RGB")

crop_dir2 = f"masked_objects_for_{img2_name}"
os.makedirs(crop_dir2, exist_ok=True)

og_width1, og_height1 = img1.size
og_width2, og_height2 = img2.size

input_tensor1 = img_tensor(img1, og_width1, og_height1)
input_tensor2 = img_tensor(img2, og_width2, og_height2)

preds1 = get_predictions(input_tensor1)
preds2 = get_predictions(input_tensor2)

objs1 = get_objs(og_width1, og_height1, preds1, img1, crop_dir1)
objs2 = get_objs(og_width2, og_height2, preds2, img2, crop_dir2)

"""
emb1 = get_clip_embeddings(objs1)
emb2 = get_clip_embeddings(objs2)

similarity_matrix = emb1 @ emb2.T

best_match_indices = torch.argmax(similarity_matrix, dim=1)
for i, j in enumerate(best_match_indices):
    print(f"Image1 object {i} ↔ Image2 object {j} | Similarity = {similarity_matrix[i, j]:.2f}")
"""