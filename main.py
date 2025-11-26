#Do not run with CTRL + ALT + N in VSCodium. Run by double clicking main.py or running "py main.py" in terminal.

import rcnn_clip
import clip
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

def swapping(first_objs, sec_objs, first_crops, sec_crops, match_list, save_dir):
    for first_objs, sec_objs in match_list.items():
        src_path = os.path.join(first_crops, first_objs)   # object from image 1
        tgt_path = os.path.join(sec_crops, sec_objs)   # its matched partner from image 2

        print(f"Swapping image {src_path} with {tgt_path}")

        src_pil = Image.open(src_path).convert("RGB")
        tgt_pil = Image.open(tgt_path).convert("RGB")

        swapped_color = rcnn_clip.color_transfer(src_pil, tgt_pil)
        swapped_histo = rcnn_clip.histo_transfer(swapped_color, tgt_pil)

        src_np = np.array(swapped_color).astype(np.float32)
        hist_np = np.array(swapped_histo).astype(np.float32)

        blended = (0.8) * src_np + 0.2 * hist_np
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        result = Image.fromarray(blended)

        result.save(os.path.join(save_dir, first_objs))

### MAIN ###

weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
detector = maskrcnn_resnet50_fpn_v2(weights=weights).eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
])

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)

# Update test images
img1_name = "wolf.png"
img2_name = "kitty.png"

img1 = Image.open(img1_name).convert("RGB")
  
crop_dir1 = f"masked_objects_for_{img1_name}"
os.makedirs(crop_dir1, exist_ok=True)

img2 = Image.open(img2_name).convert("RGB")

#rcnn_clip.CopyImageOntoBackground(img2, img1, "WolfAndKitty.png");

crop_dir2 = f"masked_objects_for_{img2_name}"
os.makedirs(crop_dir2, exist_ok=True)

og_width1, og_height1 = img1.size
og_width2, og_height2 = img2.size

input_tensor1 = transform(img1).unsqueeze(0)
input_tensor2 = transform(img2).unsqueeze(0)

preds1 = rcnn_clip.get_predictions(input_tensor1, detector)
preds2 = rcnn_clip.get_predictions(input_tensor2, detector)

obj_list1, labels1 = rcnn_clip.get_objs(og_width1, og_height1, preds1, img1, crop_dir1, weights)
obj_list2, labels2 = rcnn_clip.get_objs(og_width2, og_height2, preds2, img2, crop_dir2, weights)

# Swap object's from list 1 with list 2 and vice versa if first image has more objects
if len(obj_list1) > len(obj_list2):
    temp_list1 = obj_list1
    obj_list1 = obj_list2
    obj_list2 = temp_list1

    temp_labels1 = labels1
    labels1 = labels2
    labels2 = temp_labels1

    temp_crop_dir1 = crop_dir1
    crop_dir1 = crop_dir2
    crop_dir2 = temp_crop_dir1

    temp_img1_name = img1_name
    img1_name = img2_name
    img2_name = temp_img1_name

text_embs1 = rcnn_clip.encode_labels_as_text(labels1, device, clip_model)
text_embs2 = rcnn_clip.encode_labels_as_text(labels2, device, clip_model)

### FIND SIMILARITIES ###

similarity_list = []

for i, file1 in enumerate(obj_list1):
    # Grab image from file path (image 1)
    img_proc1 = preprocess(Image.open(os.path.join(crop_dir1, file1)).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb1 = clip_model.encode_image(img_proc1)
    emb1 /= emb1.norm(dim=-1, keepdim=True)

    for j, file2 in enumerate(obj_list2):
        # Grab image from file path (image 2)
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

similarity_list.sort(key=lambda x: x["sim_combined"], reverse=True)

print("\nTop object similarities:")
for s in similarity_list[:10]:
    print(f"{s['img1']} ({s['label1']}) ↔ {s['img2']} ({s['label2']}) | "
          f"ImageSim: {s['sim_img']:.3f} | TextSim: {s['sim_text']:.3f} | Combined: {s['sim_combined']:.3f}")

### IMAGE 1 MATCHING ###

# Final matching list in order
final_matches = {}
reverse_matches = {}

# Sets of unordered images/objects
set_img1 = set()
set_img2 = set()

# Have 2 loops, one for main order, then for reverse order
# Iterate though similarity_list
for pair in similarity_list:
    obj1 = pair["img1"]
    obj2 = pair["img2"]

    # Skip if obj1 is already matched
    if obj1 in set_img1:
        continue

    # obj2 unused -> simple match
    if obj2 not in set_img2:
        final_matches[obj1] = obj2
        reverse_matches[obj2] = obj1
        set_img1.add(obj1)
        set_img2.add(obj2)
    else:
        # obj2 already matched, so follow the chain
        final_target = rcnn_clip.resolve_chain(obj2, reverse_matches)

        final_matches[obj1] = final_target
        set_img1.add(obj1)

### IMAGE 2 MATCHING ###

# Final matching list in order
final_matches2 = {}
reverse_matches2 = {}

# Sets of unordered images/objects
set_img1_2 = set()
set_img2_2 = set()

# Iterate though similarity_list
for pair in similarity_list:
    obj1 = pair["img2"]
    obj2 = pair["img1"]

    # Skip if obj1 is already matched
    if obj1 in set_img1_2:
        continue

    # obj2 unused -> simple match
    if obj2 not in set_img2_2:
        final_matches2[obj1] = obj2
        reverse_matches2[obj2] = obj1
        set_img1_2.add(obj1)
        set_img2_2.add(obj2)
    else: # obj2 already matched -> chain following
        final_target = rcnn_clip.resolve_chain(obj2, reverse_matches)

        final_matches2[obj1] = final_target
        set_img1_2.add(obj1)

# Display top results
print("\nFinal similarities image 1:")
print(final_matches)
print("\nFinal similarities image 2:")
print(final_matches2)

swapped_img1_dir = f"swapped_img_{img1_name}"
os.makedirs(swapped_img1_dir, exist_ok=True)

print("\nSwapping color of objects:")

swapping(obj1, obj2, crop_dir1, crop_dir2, final_matches, swapped_img1_dir)

swapped_img2_dir = f"swapped_img_{img2_name}"
os.makedirs(swapped_img2_dir, exist_ok=True)

swapping(obj2, obj1, crop_dir2, crop_dir1, final_matches2, swapped_img2_dir)


# Stitching the images together. As of rn, this is all hardcoded.
os.makedirs("final_pasted_images", exist_ok=True);

finalBackgrounds = [];
finalImages = [];
finalImgNames = ["final_pasted_images/FinalKitty.png"]; #.pngs

# dogBackground = Image.open("masked_objects_for_chihuahua.png/background.png").convert("RGB");
kittyBackground = Image.open("masked_objects_for_kitty.png/background.png").convert("RGB");
# wolfBackground = Image.open("masked_objects_for_wolf.png/background.png").convert("RGB");
# finalBackgrounds = [dogBackground, kittyBackground, wolfBackground];
finalBackgrounds.append(kittyBackground);

finalKitty = Image.open("swapped_img_kitty.png/cropped_mask_cat_1.png").convert("RGB");
finalImages.append(finalKitty);

for i in range(len(finalBackgrounds)):
    rcnn_clip.CopyImageOntoBackground(finalImages[i], finalBackgrounds[i], finalImgNames[i]);