import os
import cv2
import numpy as np
from creadto.models.recon import BLASS

def run():
    root = r"D:\dump"
    files = os.listdir(os.path.join(root, "raw"))
    files = [os.path.join(root, "raw", x) for x in files]
    raw_images = []
    hlamp = BLASS()
    h = 450
    w = 300
    for i, file in enumerate(files):
        image = cv2.imread(file)
        cv2.imwrite("%05d-inputs.jpg" % i, image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        raw_images.append(image)
    raw_images = np.stack(raw_images, axis=0)
    hlamp(raw_images)
    