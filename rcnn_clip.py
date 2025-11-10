import clip
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
    cropped_object = masked_img_pil.crop(crop_area_tight).convert("RGBA")
    # Handle transparency
    background = Image.new("RGB", cropped_object.size, (255, 255, 255))  # white background
    background.paste(cropped_object, mask=cropped_object.split()[3])  # use alpha channel
    
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
    obj_labels = []
    obj_files = []

    for i in range(len(preds["boxes"])):
        label_name = weights.meta['categories'][preds["labels"][i].item()]
        mask = apply_mask_and_save_refined(
            img,
            preds["masks"][i],
            label_name,
            i + 1,
            False,
            crop_dir,
            og_width,
            og_height
        )

        if mask is not None:
            all_foreground_masks_np = np.logical_or(all_foreground_masks_np, mask)
            obj_labels.append(label_name)
            obj_files.append(f"cropped_mask_{label_name}_{i+1}.png")

    # Save background
    save_background(img, all_foreground_masks_np, og_width, og_height, crop_dir)
    # comment these 2 lines to exlude backgroung
    #obj_labels.append("unknown")
    #obj_files.append(f"background.png")

    return obj_files, obj_labels


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

def gather_imgs(crop_dir):
    file_list = []
    for item in os.listdir(crop_dir):
        full_path = os.path.join(crop_dir, item)

        if os.path.isfile(full_path):
            file_list.append(item)

    return file_list

def encode_labels_as_text(labels):
    text_tokens = clip.tokenize([f"a photo of a {lbl}" for lbl in labels]).to(device)
    with torch.no_grad():
        text_embs = clip_model.encode_text(text_tokens)
    text_embs /= text_embs.norm(dim=-1, keepdim=True)
    return text_embs

weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
detector = maskrcnn_resnet50_fpn_v2(weights=weights).eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)

# Update test images
img1_name = "kitty.png"
img2_name = "cow.png"

img1 = Image.open(img1_name).convert("RGB")

crop_dir1 = f"masked_objects_for_{img1_name}"
os.makedirs(crop_dir1, exist_ok=True)

img2 = Image.open(img2_name).convert("RGB")

crop_dir2 = f"masked_objects_for_{img2_name}"
os.makedirs(crop_dir2, exist_ok=True)

og_width1, og_height1 = img1.size
og_width2, og_height2 = img2.size

input_tensor1 = transform(img1).unsqueeze(0)
input_tensor2 = transform(img2).unsqueeze(0)

preds1 = get_predictions(input_tensor1)
preds2 = get_predictions(input_tensor2)

obj_list1, labels1 = get_objs(og_width1, og_height1, preds1, img1, crop_dir1)
obj_list2, labels2 = get_objs(og_width2, og_height2, preds2, img2, crop_dir2)

# obj_list1 = gather_imgs(crop_dir1)
# obj_list2 = gather_imgs(crop_dir2)

text_embs1 = encode_labels_as_text(labels1)
text_embs2 = encode_labels_as_text(labels2)

similarity_list = []

for i, file1 in enumerate(obj_list1):
    img_proc1 = preprocess(Image.open(os.path.join(crop_dir1, file1)).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb1 = clip_model.encode_image(img_proc1)
    emb1 /= emb1.norm(dim=-1, keepdim=True)

    for j, file2 in enumerate(obj_list2):
        img_proc2 = preprocess(Image.open(os.path.join(crop_dir2, file2)).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb2 = clip_model.encode_image(img_proc2)
        emb2 /= emb2.norm(dim=-1, keepdim=True)

        sim_img = (emb1 @ emb2.T).item()
        if (text_embs1[i] == "unknown" or text_embs2[j] == "unknown"):
            sim_text = 0
        else:
            sim_text = (text_embs1[i] @ text_embs2[j]).item()
        sim_combined = 0.6 * sim_img + 0.4 * sim_text # Adjust strength of img score vs text score

        similarity_list.append({
            "img1": file1,
            "label1": labels1[i],
            "img2": file2,
            "label2": labels2[j],
            "sim_img": sim_img,
            "sim_text": sim_text,
            "sim_combined": sim_combined
        })


# similarity_list.sort(key=lambda x: x["similarity"], reverse=True)
similarity_list.sort(key=lambda x: x["sim_combined"], reverse=True)

print("\nTop object similarities:")
for s in similarity_list[:10]:
    print(f"{s['img1']} ({s['label1']}) ↔ {s['img2']} ({s['label2']}) | "
          f"ImageSim: {s['sim_img']:.3f} | TextSim: {s['sim_text']:.3f} | Combined: {s['sim_combined']:.3f}")

"""
# Display top results
print("\nTop object similarities:")
for s in similarity_list[:10]:
    print(f"{s['img1']} ↔ {s['img2']} | Similarity: {s['similarity']:.3f}")
"""