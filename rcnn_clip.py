import clip
import cv2
import torch
from PIL import Image
from skimage.exposure import match_histograms
import numpy as np
import os

def get_predictions(image_tensor, detector):
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

def get_objs(og_width, og_height, preds, img, crop_dir, weights):
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
    obj_labels.append("background")
    obj_files.append(f"background.png")

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

def encode_labels_as_text(labels, device, clip_model):
    text_tokens = clip.tokenize([f"a photo of a {lbl}" for lbl in labels]).to(device)
    with torch.no_grad():
        text_embs = clip_model.encode_text(text_tokens)
    text_embs /= text_embs.norm(dim=-1, keepdim=True)
    return text_embs

def PatrickTestingPIL():
    #testing PIL pasting
    PIL_IMG1_NAME = "cow.png";
    PIL_IMG2_NAME = "kitty.png";

    PIL_IMG1 = Image.open(PIL_IMG1_NAME).convert("RGB");
    PIL_IMG2 = Image.open(PIL_IMG2_NAME).convert("RGB");

    PIL_IMG1.paste(PIL_IMG2);
    PIL_IMG1.save('CowAndKitty.png', quality=95);

def CopyImageOntoBackground(copyImg, backgroundImg, newImgName):
    newImg = backgroundImg.copy();    #have to force deep copy cause of Python BS
    newImg.paste(copyImg);
    newImg.save(newImgName, quality=95);


#borrows logic from the swapping() function in main.py
def Copying(firstObjPair, secondObjPair, firstCropDirectory, secondCropDirectory, matchList, saveDirectory):
    #ask Jase how matchList works and how to grab appropriate background and copy img
    
    for first_objs, sec_objs in matchList.items():
        #!this is the wrong combination, you need cropped img + background
        src_path = os.path.join(firstCropDirectory, first_objs);   # object from image 1
        tgt_path = os.path.join(secondCropDirectory, sec_objs);   # its matched partner from image 2


        print(f"Copying image {src_path} with {tgt_path}");

        img1 = Image.open(src_path).convert("RGB");
        img2 = Image.open(tgt_path).convert("RGB");

        #combinedFileName = saveDirectory + ;

        #!(copyImg, backgroundImg, newImgName)
        # CopyImageOntoBackground(img1, img2, combinedFileName);

def resolve_chain(target, reverse_matches):
    # Follow chain until reaching final unmatched object
    while target in reverse_matches:
        target = reverse_matches[target]
    return target

def color_transfer(src_pil, tgt_pil):
    # Convert PIL to cv2 LAB
    src = cv2.cvtColor(np.array(src_pil), cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt = cv2.cvtColor(np.array(tgt_pil), cv2.COLOR_RGB2LAB).astype(np.float32)

    # Compute stats
    src_mean, src_std = cv2.meanStdDev(src)
    tgt_mean, tgt_std = cv2.meanStdDev(tgt)

    src_mean = src_mean.flatten()
    src_std = src_std.flatten()
    tgt_mean = tgt_mean.flatten()
    tgt_std = tgt_std.flatten()

    # Perform Reinhard color transfer
    result = (src - src_mean) * (tgt_std / src_std) + tgt_mean

    # Clip
    result = np.clip(result, 0, 255)

    # Convert back to uint8 RGB PIL image
    result = result.astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

    return Image.fromarray(result)

# Uses scikit histogram
def histo_transfer(src_pil, tgt_pil):
    src = np.array(src_pil)
    tgt = np.array(tgt_pil)

    matched = match_histograms(src, tgt, channel_axis=-1)

    return Image.fromarray(np.uint8(matched))