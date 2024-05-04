import cv2
import numpy as np


def crop(img, center, scale, res, rot=0, dtype=np.float32):
    # Upper left point
    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1, res[1] + 1],
                            center, scale, res, invert=1)) - 1
    # size of cropped image
    #  crop_shape = [br[1] - ul[1], br[0] - ul[0]]
    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)

    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_shape = list(map(int, new_shape))
    new_img = np.zeros(new_shape, dtype=img.dtype)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]

    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    # Range to sample from original image
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]
            ] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]

    #  pixel_scale = 1.0 if new_img.max() > 1.0 else 255
    #  resample = pil_img.BILINEAR
    if not rot == 0:
        new_H, new_W, _ = new_img.shape

        rotn_center = (new_W / 2.0, new_H / 2.0)
        M = cv2.getRotationMatrix2D(rotn_center, rot, 1.0).astype(np.float32)

        new_img = cv2.warpAffine(new_img, M, tuple(new_shape[:2]),
                                 cv2.INTER_LINEAR_EXACT)
        new_img = new_img[pad:new_H - pad, pad:new_W - pad]

    output = cv2.resize(new_img, tuple(res), interpolation=cv2.INTER_LINEAR)
    return output.astype(np.float32)