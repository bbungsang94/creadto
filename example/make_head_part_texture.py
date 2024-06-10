import os
import os.path as osp
import cv2
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
    
def map_body_texture(face_texture_path, mask_path):
    face_texture = cv2.imread(face_texture_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    face_texture = np.pad(face_texture, ((0, 584), (0, 410), (0, 0)), 'constant', constant_values=0)
    mask = np.pad(mask, ((0, 584), (0, 410)), 'constant', constant_values=0)
    cv2.imwrite("padding_face.jpg", face_texture)
    cv2.imwrite("padding_mask.jpg", mask)

def merge_mask(root=r"D:\Creadto\CreadtoLibrary\creadto-model\flame\mask_images", parts=["eye_region", "eyeball", "lips"]):
    mask_files = os.listdir(root)
    observed_mask = np.zeros((512, 512))
    for mask_file in mask_files:
        if mask_file.replace(".jpg", "") not in parts:
            continue
        mask = cv2.imread(osp.join(root, mask_file), cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        observed_mask = cv2.bitwise_or(observed_mask, mask)
    
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

def modify_skin_color(mst_root=r"creadto-model\textures\MonkSkinToneScale\MST Swatches",
                      target_image=r"creadto-model\flame\mask_images\default_image.png",
                      skin_mask_path=r"creadto-model\flame\mask_images\skin.jpg"):
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
    
    # Get skin data
    now_skin = np.zeros(3)
    for i in range(3):
        panel = skin_image[:, :, i]
        min_value, max_value = monk_skin_tone_scale.min(axis=0)[i], monk_skin_tone_scale.max(axis=0)[i]
        subset = panel[(panel >= min_value) & (panel <= max_value)]
        now_skin[i] = subset.mean()
    
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
               default_path=r"./creadto-model/flame/default_texture/realistic_origin.png",
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

def run_full_cycle(root):
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
        
if __name__ == "__main__":
    merge_mask()
    # make_contour_mask()
    # make_images(mask_root=r"D:\Creadto\CreadtoLibrary\creadto-model\flame\mask_images", origin_root=r"D:\Creadto\CreadtoLibrary\creadto-model\flame\default_texture")
    # map_body_texture(r"D:\Creadto\CreadtoLibrary\output\only_face.jpg", r"D:\Creadto\CreadtoLibrary\output\inference_mask.jpg")
    # merge_face_default(face_path=r"D:\dump\temp\result_head-0th.png", mask_root=r"D:\Creadto\CreadtoLibrary\creadto-model\flame\mask_images")
    # map_body_texture(face_texture_path=r"./merged_image.png", mask_path=r"D:\Creadto\CreadtoLibrary\creadto-model\flame\mask_images\inference_mask.jpg")
    # modify_skin_color()
    #run_full_cycle(root=r"D:/dump/head_model_test")
    # run_cut_only_head_image()