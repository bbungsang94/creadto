import copy
import os
import os.path as osp
from typing import Dict, List
import cv2
import torch
import numpy as np
from example.run_hlamp import image_to_flaep

def make_images(mask_root, origin_root):
    files = os.listdir(mask_root)
    origin_image = cv2.imread(osp.join(origin_root, "realistic_origin.png"), cv2.IMREAD_COLOR)
    origin_image = cv2.resize(origin_image, (512, 512))
    for file in files:
        mask_image = cv2.imread(osp.join(mask_root, file), cv2.IMREAD_GRAYSCALE)
        _, mask_image = cv2.threshold(mask_image, 128, 255, cv2.THRESH_BINARY)
        masked = cv2.bitwise_and(origin_image, origin_image, mask=mask_image)
        cv2.imwrite(osp.join(origin_root, file), masked)

def deca_to_head():
    from torchvision.transforms.functional import to_pil_image
    test, model = image_to_flaep(root=r"D:\dump\temp")
    model.save_obj(osp.join(r"D:\dump\temp", "result_head.obj"), test)
    uv_textures = test['uv_texture_gt']
    for i, uv_texture in enumerate(uv_textures):
        image = to_pil_image(uv_texture)
        image.save(osp.join(r"D:\dump\uv_textures", "%d-th uv_textures.jpg" % i))

def merged_images(face_image_path, default_root, mask_root, parts=["eye_region", "nose", "lips", "left_eyeball", "right_eyeball"], departs=[]):
    face_image = cv2.imread(face_image_path)
    face_image = cv2.resize(face_image, (512, 512))
    default_image = cv2.imread(osp.join(default_root, "origin.jpg"))
    mask_files = os.listdir(mask_root)
    mask_image = np.zeros((512, 512), dtype=np.uint8)
    for mask_file in mask_files:
        if mask_file.replace(".jpg", "") not in parts:
            continue
        mask = cv2.imread(osp.join(mask_root, mask_file), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        mask_image = cv2.add(mask_image, mask)
    for mask_file in mask_files:
        if mask_file.replace(".jpg", "") not in departs:
            continue
        mask = cv2.imread(osp.join(mask_root, mask_file), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        mask_image = cv2.subtract(mask_image, mask)
    only_face_image = cv2.bitwise_and(face_image, face_image, mask=mask_image)
    default_image = cv2.bitwise_and(default_image, default_image, mask=255 - mask_image)
    cv2.imwrite("mask.jpg", mask_image)
    cv2.imwrite("only_face.jpg", only_face_image)
    cv2.imwrite("default.jpg", default_image)
    cv2.imwrite("merged_image.jpg", cv2.add(default_image, only_face_image))
    
def map_body_texture(head_albedo: torch.Tensor, body_albedo: torch.Tensor):
    """_summary_
    Args:
        head_albedo (torch.Tensor): shape (c, 512, 512)
        body_albedo (torch.Tensor): shape (c, 1024, 1024)
    """
    device = head_albedo.device
    dtype = head_albedo.dtype
    model_root = r"creadto-model"
    bridge = np.load(osp.join(model_root, "flame", "flame2smplx_tex_1024.npy"), allow_pickle=True, encoding = 'latin1').item()

    x_coords = torch.tensor(bridge['x_coords'], dtype=torch.int32, device=device, requires_grad=False)
    y_coords = torch.tensor(bridge['y_coords'], dtype=torch.int32, device=device, requires_grad=False)
    target_pixel_ids = torch.tensor(bridge['target_pixel_ids'], dtype=torch.int32, device=device, requires_grad=False)
    source_uv_points = torch.tensor(bridge['source_uv_points'], dtype=torch.float32, device=device, requires_grad=False)
    source_tex_coords = torch.zeros((source_uv_points.shape[0], source_uv_points.shape[1]), dtype=torch.int32, device=device, requires_grad=False)
    source_tex_coords[:, 0] = torch.clamp(head_albedo.shape[1]*(1.0-source_uv_points[:,1]), 0.0, head_albedo.shape[1]).type(torch.IntTensor)
    source_tex_coords[:, 1] = torch.clamp(head_albedo.shape[2]*(source_uv_points[:,0]), 0.0, head_albedo.shape[2]).type(torch.IntTensor)

    body_albedo[:, y_coords[target_pixel_ids], x_coords[target_pixel_ids]] = head_albedo[:, source_tex_coords[:,0], source_tex_coords[:,1]]
    albedo = copy.deepcopy(body_albedo)
    return albedo
        
def merge_mask(root=r"D:\Creadto\CreadtoLibrary\creadto-model\flame\mask_images", parts=["left_eye_region", "right_eye_region", "left_eyeball", "right_eyeball", "lips"]):
    mask_files = os.listdir(root)
    observed_mask = np.zeros((512, 512), dtype=np.uint8)
    for mask_file in mask_files:
        if mask_file.replace(".jpg", "") not in parts:
            continue
        mask = cv2.imread(osp.join(root, mask_file), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        observed_mask = cv2.add(observed_mask, mask)
    
    cv2.imwrite("ovserved_mask.jpg", observed_mask)
    
def merge_face_default(face_path, mask_root):
    face_image = cv2.imread(face_path, cv2.IMREAD_COLOR)
    face_image = cv2.resize(face_image, (512, 512))
    observed_mask = cv2.imread(osp.join(mask_root, "observed_mask.jpg"), cv2.IMREAD_GRAYSCALE)
    _, observed_mask = cv2.threshold(observed_mask, 128, 255, cv2.THRESH_BINARY)
    face_image = cv2.bitwise_and(face_image, face_image, mask=observed_mask)
    
    default_image = cv2.imread(osp.join(mask_root, "default_image.png"), cv2.IMREAD_COLOR)
    default_image = cv2.resize(default_image, (512, 512))
    default_image = cv2.bitwise_and(default_image, default_image, mask=255-observed_mask)
    
    result_image = cv2.add(default_image, face_image)
    resized_image = cv2.resize(result_image, (256, 256))
    cv2.imwrite("merged_image.png", result_image)
    cv2.imwrite("resized_merged_image.png", resized_image)

def make_contour_mask(root=r"creadto-model/flame/mask_images/", parts=["eye_region", "nose", "lips", "observed_mask"]):
    mask_files = os.listdir(root)
    face_image = np.zeros((512, 512), dtype=np.uint8)
    for mask_file in mask_files:
        if mask_file.replace(".jpg", "") not in parts:
            continue
        mask = cv2.imread(osp.join(root, mask_file), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        face_image = cv2.add(face_image, mask)
    
    contours, hier = cv2.findContours(face_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    board = np.zeros_like(face_image)
    cv2.drawContours(board, contours, -1, (255, 255, 255), 6)
    cv2.imwrite("face_contour.jpg", board)
        
def make_skin_mask(root=r"creadto-model\flame\mask_images", departs=["forehead", "eye_region", "nose", "lips", "scalp"]):
    skin_mask = cv2.imread(osp.join(root, "face.jpg"), cv2.IMREAD_GRAYSCALE)
    _, skin_mask = cv2.threshold(skin_mask, 128, 255, cv2.THRESH_BINARY)
    for mask_file in os.listdir(root):
        if mask_file.replace(".jpg", "") not in departs:
            continue
        demask = cv2.imread(osp.join(root, mask_file), cv2.IMREAD_GRAYSCALE)
        _, demask = cv2.threshold(demask, 128, 255, cv2.THRESH_BINARY)
        skin_mask = cv2.subtract(skin_mask, demask)
    cv2.imwrite("skin.jpg", skin_mask)

def modify_skin_color(mst_root=r"creadto-model\textures\MSTScale\MST Swatches",
                      target_image=r"creadto-model\flame\mask_images\default_image.png",
                      skin_mask_path=r"creadto-model\flame\mask_images\mono-body-mask.png"):
    #Load MST
    monk_skin_tone_scale = []
    skin_files = os.listdir(mst_root)
    for skin_file in skin_files:
        skin = cv2.imread(osp.join(mst_root, skin_file))
        monk_skin_tone_scale.append([skin[0, 0, 0], skin[0, 0, 1], skin[0, 0, 2]])
    monk_skin_tone_scale = np.array(monk_skin_tone_scale)
    # Prepare Images
    origin_image = cv2.imread(target_image)
    skin_mask = cv2.imread(skin_mask_path, cv2.IMREAD_GRAYSCALE)
    skin_mask = cv2.resize(skin_mask, (origin_image.shape[0], origin_image.shape[1]))
    _, skin_mask = cv2.threshold(skin_mask, 128, 255, cv2.THRESH_BINARY)
    skin_image = cv2.bitwise_and(origin_image, origin_image, mask=skin_mask)
    
    cv2.imwrite("skin_image.jpg", skin_image)
    # Get skin data
    now_skin = np.zeros(3)
    for i in range(3):
        panel = skin_image[:, :, i]
        min_value, max_value = monk_skin_tone_scale.min(axis=0)[i], monk_skin_tone_scale.max(axis=0)[i]
        min_value, max_value = 20, 255
        subset = panel[(panel >= min_value) & (panel <= max_value)]
        now_skin[i] = np.median(subset)
    
    # Find skin tone
    error = abs(monk_skin_tone_scale - now_skin)
    tone_index = np.argmin(error.sum(axis=1))
    diff = monk_skin_tone_scale[tone_index] - now_skin
    changed_image = np.clip(origin_image + diff, 0, 255)
    cv2.imwrite("fit_skin.png", changed_image)
    
    # Explore all skin tones
    for i, filename in enumerate(skin_files):
        diff = monk_skin_tone_scale[i] - now_skin
        changed_image = np.clip(origin_image + diff, 0, 255)
        cv2.imwrite("default-texture-%s" % filename, changed_image)


def match_skin(face_path=r"./output/deca_output.png",
               default_path=r"./creadto-model/textures/MSTScale/Samples/default_head (1).png",
               skin_mask_path=r"./creadto-model/flame/mask_images/skin.jpg",
               observed_mask_path=r"./creadto-model/flame/mask_images/observed_mask.jpg",
               contour_path=r"./creadto-model/flame/mask_images/face_contour.jpg",
               mst_root=r"creadto-model/textures/MSTScale/MST Swatches",
               prefix="",
               save_log=True):
    #Load MST
    monk_skin_tone_scale = []
    skin_files = os.listdir(mst_root)
    for skin_file in skin_files:
        skin = cv2.imread(osp.join(mst_root, skin_file))
        monk_skin_tone_scale.append([skin[0, 0, 0], skin[0, 0, 1], skin[0, 0, 2]])
    monk_skin_tone_scale = np.array(monk_skin_tone_scale)
    
    deca_image = cv2.imread(face_path)
    deca_image = cv2.resize(deca_image, (512, 512))
    if save_log:
        cv2.imwrite(prefix + "input_image.png", deca_image) 
    default_image = cv2.imread(default_path)
    
    skin_mask = cv2.imread(skin_mask_path, cv2.IMREAD_GRAYSCALE)
    deca_skin_mask = cv2.resize(skin_mask, (deca_image.shape[0], deca_image.shape[1]))
    _, deca_skin_mask = cv2.threshold(deca_skin_mask, 128, 255, cv2.THRESH_BINARY)
    deca_skin_image = cv2.bitwise_and(deca_image, deca_image, mask=deca_skin_mask)
    if save_log:
        cv2.imwrite(prefix + "pre_skin_data.png", deca_skin_image)    
    
    default_skin_mask = cv2.resize(skin_mask, (default_image.shape[0], default_image.shape[1]))
    _, default_skin_mask = cv2.threshold(default_skin_mask, 128, 255, cv2.THRESH_BINARY)
    default_skin_image = cv2.bitwise_and(default_image, default_image, mask=default_skin_mask)
    if save_log:
        cv2.imwrite(prefix + "pre_default.png", default_skin_image) 
    
    # Get skin data
    image_skin = np.zeros(3)
    default_skin = np.zeros(3)
    deca_subset = np.ones((deca_skin_image.shape[0], deca_skin_image.shape[0]), dtype=bool)
    default_subset = np.ones((default_skin_image.shape[0], default_skin_image.shape[0]), dtype=bool)
    for i in range(3):
        deca_panel = deca_skin_image[:, :, i]
        min_value, max_value = monk_skin_tone_scale.min(axis=0)[i] - 10, monk_skin_tone_scale.max(axis=0)[i] + 60
        deca_subset &= (deca_panel >= min_value) & (deca_panel <= max_value)
        
        default_panel = default_skin_image[:, :, i]
        default_subset &= (default_panel >= min_value) & (default_panel <= max_value)
    if save_log:
        cv2.imwrite(prefix + "input_skin_detected.png", deca_subset.astype(np.float32) * 255)
        cv2.imwrite(prefix + "default_skin_detected.png", default_subset.astype(np.float32) * 255)
    
    for i in range(3):
        deca_panel = deca_skin_image[:, :, i] * deca_subset
        default_panel = default_skin_image[:, :, i] * default_subset
        deca_filter = deca_panel[deca_panel != 0]
        default_filter = default_panel[default_panel != 0]
        image_skin[i] = deca_filter.mean()
        default_skin[i] = default_filter.mean()
            
    # Find skin tone and match
    error = abs(monk_skin_tone_scale - image_skin)
    tone_index = np.argmin(error.sum(axis=1))
    deca_diff = monk_skin_tone_scale[tone_index] - image_skin
    default_diff = monk_skin_tone_scale[tone_index] - default_skin
    deca_subset = np.ones((deca_image.shape[0], deca_image.shape[0]), dtype=bool)
    default_subset = np.ones((default_image.shape[0], default_image.shape[0]), dtype=bool)
    print(skin_files[tone_index])
    for i in range(3):
        min_value, max_value = monk_skin_tone_scale.min(axis=0)[i] - 10, monk_skin_tone_scale.max(axis=0)[i] + 60
        deca_panel = deca_image[:, :, i]
        default_panel = default_image[:, :, i]
        deca_subset &= (deca_panel >= min_value) & (deca_panel <= max_value)
        default_subset &= (default_panel >= min_value) & (default_panel <= max_value)
    if save_log:
        cv2.imwrite(prefix + "input_skin_applied.png", deca_subset.astype(np.float32) * 255)
        cv2.imwrite(prefix + "default_skin_applied.png", default_subset.astype(np.float32) * 255)
    for i in range(3):
        deca_image[:, :, i] = np.clip(deca_subset * deca_diff[i] + deca_image[:, :, i], 0, 255)
        default_image[:, :, i] = np.clip(default_subset * default_diff[i] + default_image[:, :, i], 0, 255)

    if save_log:
        cv2.imwrite(prefix + "post_skin_data.png", deca_image) 
        cv2.imwrite(prefix + "post_default.png", default_image) 
    
    # Merge image
    observed_mask = cv2.imread(observed_mask_path, cv2.IMREAD_GRAYSCALE)
    _, observed_mask = cv2.threshold(observed_mask, 128, 255, cv2.THRESH_BINARY)
    face_image = cv2.bitwise_and(deca_image, deca_image, mask=observed_mask)
    default_image = cv2.resize(default_image, (observed_mask.shape[0], observed_mask.shape[1]))
    default_image = cv2.bitwise_and(default_image, default_image, mask=255-observed_mask)
    merged_image = cv2.add(default_image, face_image)
    if save_log:
        cv2.imwrite(prefix + "cut_default.png", default_image)
        cv2.imwrite(prefix + "cut_face.png", face_image)
        cv2.imwrite(prefix + "merged_image.png", cv2.resize(merged_image, (256, 256)))
    
    # Blur contour
    contour_mask = cv2.imread(contour_path, cv2.IMREAD_GRAYSCALE)
    _, observed_mask = cv2.threshold(observed_mask, 128, 255, cv2.THRESH_BINARY)
    blurred_image = cv2.GaussianBlur(merged_image, (21, 21), 9, 9)
    blurred_image = cv2.blur(merged_image, (25, 25))
    if save_log:
        cv2.imwrite(prefix + "final_blur.png", blurred_image)
    mask_3ch = cv2.cvtColor(contour_mask, cv2.COLOR_GRAY2RGB)
    final_image = np.where(mask_3ch == 255, blurred_image, merged_image)
    final_image = cv2.resize(final_image, (256, 256))
    if save_log:
        cv2.imwrite(prefix + "final_output.png", final_image)
    return cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

def run_displace_cycle(root):
    import torch
    from PIL import Image
    from torchvision.transforms import ToTensor, ToPILImage
    from creadto.models.recon import DetailFaceModel
    from creadto.utils.vision import remove_light
    flaep = DetailFaceModel()
    trans = ToTensor()
    to_pil = ToPILImage()

    files = os.listdir(osp.join(root, "head_images"))

    raw_images = []
    for file in files:
        image_bgr = cv2.imread(osp.join(root, "head_images", file))
        # removal = cv2.resize(image_bgr, (512, 512))
        removal = remove_light(image_bgr)
        cv2.imwrite(osp.join(root, "removal_images", file), removal)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        raw_images.append(trans(image_rgb))
    raw_images = torch.stack(raw_images, dim=0)
    
    result = flaep.decode(raw_images.cuda())
    image_texture = result["uv_texture_gt"]
    for i, face_image in enumerate(image_texture):
        pil_image: Image = to_pil(face_image)
        pil_image.save("face_image.png")
        image_texture[i] = trans(match_skin(face_path=r"./face_image.png", prefix="%d-" % i, save_log=False))
        
    result = flaep.decode(raw_images.cuda(), external_tex=image_texture)
    image_texture = result["uv_texture_gt"]
    for i, face_image in enumerate(image_texture):
        pil_image: Image = to_pil(face_image)
        pil_image.save("face_image.png")
        image_texture[i] = trans(match_skin(face_path=r"./face_image.png", prefix="%d-" % i, save_log=True))
    flaep.reconstructor.save_obj(osp.join(r"D:\dump\head_model_test\output_models", "head.obj"), result)

def run_cut_only_head_image(root=r"D:\dump\head_model_test\input_images"):
    import torch
    from PIL import Image
    from torchvision.transforms import ToTensor, ToPILImage
    from creadto.models.recon import DetailFaceModel
    flaep = DetailFaceModel()
    trans = ToTensor()
    to_pil = ToPILImage()

    files = os.listdir(root)

    raw_images = []
    for file in files:
        image = Image.open(osp.join(root, file))
        raw_images.append(trans(image))
    result, process = flaep.encode(raw_images)
    for i, face_image in enumerate(result):
        pil_image: Image = to_pil(face_image)
        pil_image.save(osp.join(root, "head-%s" % files[i]))

def resize_to(root=r"D:\dump\head_model_test\input_images\imgs", target=(512, 512)):
    files = os.listdir(root)
    files = [x for x in files if ".jpeg" in x or ".jpg" in x or ".png" in x]
    for file in files:
        image = cv2.imread(osp.join(root, file))
        image = cv2.resize(image, target)
        cv2.imwrite(osp.join(root, "%d-%s" %(target[0], file)), image)

def enhance_face_skin(images,
                      mst_root=r"D:\Creadto\CreadtoLibrary\creadto-model\textures\MSTScale",
                      iteration=1):
    import facer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #Load MST
    mst = np.load(osp.join(mst_root, "MSTScaleRGB.npy"))
    mst = np.delete(mst, len(mst)-1, 0)
    mst = np.delete(mst, len(mst)-1, 0)
    # define categories of face parser
    categories = {"background": 0, "neck": 1, "skin": 2, "cloth": 3,
                  "left_ear": 4, "right_ear": 5, "left_eyebrow": 6, "right_eyebrow": 7,
                  "left_eye": 8, "right_eye": 9, "nose": 10, "mouth": 11,
                  "lower_lip": 12, "upper_lip": 13, "hair": 14, "sunglasses": 15,
                  "hat": 16, "earring": 17, "necklace": 18}
    
    face_detector = facer.face_detector('retinaface/mobilenet', device=device, model_path=r"./creadto-model/mobilenet0.25_Final.pth")
    face_parser = facer.face_parser('farl/celebm/448', model_path=r"./creadto-model/face_parsing.farl.celebm.main_ema_181500_jit.pt", device=device) # optional "farl/lapa/448"
    with torch.inference_mode():
        faces = face_detector(images)
        faces = face_parser(images, faces)
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    vis_seg_probs = seg_probs.argmax(dim=1).float()/len(categories)*255
    vis_img = vis_seg_probs.sum(0, keepdim=True)
    facer.show_bhw(vis_img)
    facer.show_bchw(facer.draw_bchw(images, faces))

    face_masks = seg_probs[:, categories['skin']] + seg_probs[:, categories['nose']]
    mean_values = []
    for image, face_mask in zip(images, face_masks):
        vis = image * face_mask.expand_as(image)
        facer.show_bchw(vis[None, :, :])
        
        mask = torch.ones((image.shape[1], image.shape[2]), device=device, dtype=torch.bool)
        for i in range(3):
            mono_albedo = vis[i]
            min_value, max_value = mst.min(axis=0)[i], mst.max(axis=0)[i] + 60
            mask &= (mono_albedo >= min_value) & (mono_albedo <= max_value)
            temp = mask.expand_as(vis).type(face_mask.dtype)
            # facer.show_bchw(temp[None, :, :] * 255)
        
        filtered = vis * (mask.expand_as(vis).type(face_mask.dtype))
        facer.show_bchw(filtered[None, :, :])
        
        mean_value = torch.zeros(3)        
        for i in range(3):
            mono_albedo = vis[i]
            selected_values = mono_albedo[mask]
            if selected_values.numel() > 0:
                mean_value[i] = selected_values.median()
            else:
                mean_value[i] = float('nan')  # 값이 없을 경우 NaN
        diff = torch.tensor(mst, dtype=mean_value.dtype, device=mean_value.device) - mean_value 
        print(diff)
        tone_index = torch.argmin(abs(diff).sum(dim=1))
        # mst_value = mst[tone_index]
        mst_value = mean_value
        print(mst_value)
        flat_skin = torch.zeros_like(vis)
        mean_skin = torch.zeros_like(vis)
        for i in range(3):
            flat_skin[i, :, :] = face_mask * mst_value[i]
        facer.show_bchw(flat_skin[None, :, :, :])
        mean_skin = (vis + flat_skin) / 2.0
        for i in range(iteration - 1):
            mean_skin = (mean_skin + flat_skin) / 2.0
        facer.show_bchw(mean_skin[None, :, :, :])
        
        face_filter = face_mask.expand_as(image)
        enhanced = image * (1 - face_filter) + mean_skin * face_filter
        facer.show_bchw(enhanced[None, :, :, :])
        mean_values.append(mean_value)
    return torch.stack(mean_values).to(images.device)

def merge_mono_albedos():
    import torchvision
    import torchvision.io as io
    import torchvision.utils as util
    root = r"D:\Creadto\CreadtoLibrary\creadto-model\textures\MSTScale\Samples"
    body = "mono_body-adult.png"
    head = "mono_head2.png"
    resizer512 = torchvision.transforms.Resize((512, 512))
    resizer1024 = torchvision.transforms.Resize((1024, 1024))
    body_albedo = io.read_image(osp.join(root, body))
    #body_albedo = resizer1024(body_albedo)
    head_albedo = io.read_image(osp.join(root, head))
    #head_albedo = resizer512(head_albedo)
    
    albedo = map_body_texture(head_albedo[:3], body_albedo[:3])
    pil_image = torchvision.transforms.functional.to_pil_image(albedo)
    pil_image.save("full_albedo.png")

def make_weighted_masks(mono_body_albedo_path=r"D:\Creadto\CreadtoLibrary\creadto-model\textures\MSTScale\Samples\mono_body-masks.png"):
    body_albedo = cv2.imread(mono_body_albedo_path)
    body_albedo = body_albedo.astype(np.float32)
    green_channel = copy.deepcopy(body_albedo[:, :, 1])
    green_channel = green_channel - ((body_albedo[:, :, 0] + body_albedo[:, :, 2]) / 2.)
    cv2.imwrite("weighted_green_mask.png", green_channel)
    
    red_channel = copy.deepcopy(body_albedo[:, :, -1])
    red_channel = red_channel - ((body_albedo[:, :, 0] + body_albedo[:, :, 1]) / 2.)
    cv2.imwrite("weighted_red_mask.png", red_channel)
    pass

def extract_sclera(head_image_paths=r"D:\dump\sample\head_images"):
    import facer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # define categories of face parser
    categories = {"background": 0, "neck": 1, "skin": 2, "cloth": 3,
                  "left_ear": 4, "right_ear": 5, "left_eyebrow": 6, "right_eyebrow": 7,
                  "left_eye": 8, "right_eye": 9, "nose": 10, "mouth": 11,
                  "lower_lip": 12, "upper_lip": 13, "hair": 14, "sunglasses": 15,
                  "hat": 16, "earring": 17, "necklace": 18}
    
    files = os.listdir(head_image_paths)
    images = []
    for file in files:
        image = facer.hwc2bchw(facer.read_hwc(osp.join(head_image_paths, file))).to(device=device)
        images.append(image)
    images = torch.concat(images).to(device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device, model_path=r"./creadto-model/mobilenet0.25_Final.pth")
    face_parser = facer.face_parser('farl/celebm/448', model_path=r"./creadto-model/face_parsing.farl.celebm.main_ema_181500_jit.pt", device=device) # optional "farl/lapa/448"
    with torch.inference_mode():
        faces = face_detector(images)
        faces = face_parser(images, faces)
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    vis_seg_probs = seg_probs.argmax(dim=1).float()/len(categories)*255
    vis_img = vis_seg_probs.sum(0, keepdim=True)
    facer.show_bhw(vis_img)
    facer.show_bchw(facer.draw_bchw(images, faces))

    eye_masks = seg_probs[:, categories['left_eye']] + seg_probs[:, categories['right_eye']]
    for image, face_mask in zip(images, eye_masks):
        vis = image * face_mask.expand_as(image)
        facer.show_bchw(vis[None, :, :])
        
        mask = torch.ones((image.shape[1], image.shape[2]), device=device, dtype=torch.bool)
        for i in range(3):
            mono_albedo = vis[i]
            min_value, max_value = 0, 170
            mask &= (mono_albedo >= min_value) & (mono_albedo <= max_value)
            temp = mask.expand_as(vis).type(face_mask.dtype)
            # facer.show_bchw(temp[None, :, :] * 255)
        
        filtered = vis * (mask.expand_as(vis).type(face_mask.dtype))
        facer.show_bchw(filtered[None, :, :])
        
        mean_values = torch.zeros(3)        
        for i in range(3):
            mono_albedo = vis[i]
            selected_values = mono_albedo[mask]
            if selected_values.numel() > 0:
                mean_values[i] = selected_values.median()
            else:
                mean_values[i] = float('nan')  # 값이 없을 경우 NaN
        # mst_value = mst[tone_index]
        mst_value = mean_values
        print(mst_value)
        flat_skin = torch.zeros_like(vis)
        mean_skin = torch.zeros_like(vis)
        for i in range(3):
            flat_skin[i, :, :] = face_mask * mst_value[i]
        facer.show_bchw(flat_skin[None, :, :, :])
        mean_skin = (vis + flat_skin) / 2.0
        facer.show_bchw(mean_skin[None, :, :, :])
        
        face_filter = face_mask.expand_as(image)
        enhanced = image * (1 - face_filter) + mean_skin * face_filter
        facer.show_bchw(enhanced[None, :, :, :])

# region: be included methods of NakedHuman class
def get_parts_colour(images: torch.Tensor, parts: List[str], min_thrd = 5, max_thrd = 170) -> Dict[str, object]:
    """From head image (b, 3, 224, 224), extract eye(iris) colour without sclera.

    Args:
        images (torch.Tensor): (b, 3, 224, 224) head image [0, 255]
        min_thrd (int, optional): for mask of sclera. Defaults to 0.
        max_thrd (int, optional): _for mask of sclera. Defaults to 170.

    Returns:
        Dict[str, object]: Dict of operation results such as input images, processed images, colour values, and file names
    """
    import facer
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # define categories of face parser
    categories = {"background": 0, "neck": 1, "skin": 2, "cloth": 3,
                  "left_ear": 4, "right_ear": 5, "left_eyebrow": 6, "right_eyebrow": 7,
                  "left_eye": 8, "right_eye": 9, "nose": 10, "mouth": 11,
                  "lower_lip": 12, "upper_lip": 13, "hair": 14, "sunglasses": 15,
                  "hat": 16, "earring": 17, "necklace": 18}
    
    face_detector = facer.face_detector('retinaface/mobilenet', device=device, model_path=r"./creadto-model/mobilenet0.25_Final.pth")
    face_parser = facer.face_parser('farl/celebm/448', model_path=r"./creadto-model/face_parsing.farl.celebm.main_ema_181500_jit.pt", device=device) # optional "farl/lapa/448"
    names = images['names']
    images = images['images']
    with torch.inference_mode():
        faces = face_detector(images)
        faces = face_parser(images, faces)
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    vis_seg_probs = seg_probs.argmax(dim=1).float()/len(categories)*255
    vis_img = vis_seg_probs.sum(0, keepdim=True)
    facer.show_bhw(vis_img)
    facer.show_bchw(facer.draw_bchw(images, faces))

    masks = torch.zeros((seg_probs[:, 0].shape), device=seg_probs.device, dtype=seg_probs.dtype)
    for part in parts:
        masks += seg_probs[:, categories[part]]
    mean_values = []
    for image, face_mask in zip(images, masks):
        vis = image * face_mask.expand_as(image)
        # facer.show_bchw(vis[None, :, :])
        
        mask = torch.ones((image.shape[1], image.shape[2]), device=device, dtype=torch.bool)
        for i in range(3):
            mono_albedo = vis[i]
            min_value, max_value = min_thrd, max_thrd
            mask &= (mono_albedo >= min_value) & (mono_albedo <= max_value)
        
        filtered = vis * (mask.expand_as(vis).type(face_mask.dtype))
        facer.show_bchw(filtered[None, :, :])
        
        mean_value = torch.zeros(3)        
        for i in range(3):
            mono_albedo = vis[i]
            selected_values = mono_albedo[mask]
            if selected_values.numel() > 0:
                mean_value[i] = selected_values.median()
            else:
                mean_value[i] = float('nan')  # 값이 없을 경우 NaN
        mean_values.append(mean_value)    
    print(mean_values)

    return {
        'input images': images,
        'face categories': categories,
        'mean_values': torch.stack(mean_values)
    }

def change_skin_color(target_rgbs: torch.Tensor, min_thrd=20, max_thrd=160):
    """_summary_

    Args:
        target_rgbs (torch.Tensor): (b, 3))
    """
    from creadto.utils.io import load_image
    converted = []
    for rgb in target_rgbs:
        # Find skin tone
        # base_basis = base_mask.unsqueeze(dim=0).expand_as(skin) * (base_diff - basis_median).view(-1, 1, 1)
        # fetched_skin = torch.clamp(skin + face_basis, 0.0, 1.0)
        image = load_image(r"creadto-model\textures\MSTScale\Samples\mono_body-masks.png")[0]
        image = image.type(torch.FloatTensor).to(rgb.device)
        diff = rgb - torch.tensor([205., 205., 205.]).to(rgb.device)
        painted = torch.clamp(image + diff.view(-1, 1, 1), 0, 255)
        converted.append(painted)
    return torch.stack(converted)

def paint_with_mask(images: torch.Tensor, masks: torch.Tensor, colors: torch.Tensor):
    """_summary_

    Args:
        images (torch.Tensor): (b, c, w, h)
        masks (torch.Tensor): (b, w, h)
        color (torch.Tensor): (b, 3)
    """
    painted_images = []
    colors = colors.to(images.device)
    for image, mask, color in zip(images, masks, colors):
        partial = color.view(-1, 1, 1) * mask.expand_as(image)
        painted = image * (1-mask.expand_as(image)) + partial
        painted_images.append(painted)
    return torch.stack(painted_images)
  
def run_paint_cycle(root: str = r"D:\dump\sample\head_images",
                    mask_root: str = r"D:\Creadto\CreadtoLibrary\creadto-model\textures\MSTScale\Samples") -> Dict[str, object]:
    import torchvision
    # load images
    from creadto.utils.io import load_images, load_image
    device = "cuda:0"
    images = load_images(root, device)
    eye_dict = get_parts_colour(images, ['left_eye', 'right_eye'])
    lip_dict = get_parts_colour(images, ['lower_lip', 'upper_lip'], max_thrd=255)
    eyebrow_dict = get_parts_colour(images, ['left_eyebrow', 'right_eyebrow'], max_thrd=255)
    skin_value = enhance_face_skin(images)
    
    # Paint skin-tone
    colored_skin = change_skin_color(target_rgbs=skin_value)
    for i, image in enumerate(colored_skin):
        pil_image = torchvision.transforms.functional.to_pil_image(image / 255.)
        pil_image.save("%d-th colored skin.png" % i)
    # Paint iris
    iris_masks = load_image(osp.join(mask_root, "weighted_green_mask.png"), mono=True)
    iris_masks = iris_masks.type(torch.FloatTensor) / 255.
    iris_masks = iris_masks.expand(colored_skin[:, 0].shape)
    iris_masks = iris_masks.to(colored_skin.device)
    iris_painted = paint_with_mask(colored_skin, iris_masks, eye_dict['mean_values'])
    for i, image in enumerate(iris_painted):
        pil_image = torchvision.transforms.functional.to_pil_image(image / 255.)
        pil_image.save("%d-th iris skin.png" % i)
    # Paint lips
    lip_masks = load_image(osp.join(mask_root, "weighted_red_mask.png"), mono=True)
    lip_masks = lip_masks.type(torch.FloatTensor) / 255.
    lip_masks = lip_masks.expand(iris_painted[:, 0].shape)
    lip_masks = lip_masks.to(colored_skin.device)
    lip_painted = paint_with_mask(iris_painted, lip_masks, lip_dict['mean_values'])
    for i, image in enumerate(lip_painted):
        pil_image = torchvision.transforms.functional.to_pil_image(image / 255.)
        pil_image.save("%d-th lip skin.png" % i)
    # Draw eyebrows
    pass
# endregion


if __name__ == "__main__":
    # merge_mask()
    # make_contour_mask()
    # make_images(mask_root=r"D:\Creadto\CreadtoLibrary\creadto-model\flame\mask_images", origin_root=r"D:\Creadto\CreadtoLibrary\creadto-model\flame\default_texture")
    # map_body_texture(r"D:\Creadto\CreadtoLibrary\output\only_face.jpg", r"D:\Creadto\CreadtoLibrary\output\inference_mask.jpg")
    # merge_face_default(face_path=r"D:\dump\temp\result_head-0th.png", mask_root=r"D:\Creadto\CreadtoLibrary\creadto-model\flame\mask_images")
    # map_body_texture(face_texture_path=r"./merged_image.png", mask_path=r"D:\Creadto\CreadtoLibrary\creadto-model\flame\mask_images\inference_mask.jpg")
    modify_skin_color(target_image=r"creadto-model\textures\MSTScale\Samples\mono_body_8k.png",
                      skin_mask_path=r"creadto-model\flame\mask_images\mono-body-mask.png")
    # run_full_cycle(root=r"D:/dump/sample")
    # run_cut_only_head_image()
    #dummy()
    #enhance_face_skin()
    run_paint_cycle()
    # make_weighted_masks()