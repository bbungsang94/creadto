import copy
import math
from typing import Union, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from torchvision.transforms import transforms

from external.mediapipe.convention import get_478_indexes, get_68_indexes, Mapper


class BasicFacialLandmarker:
    def __init__(self, model):
        self.model = model.cpu()
        self.n_markers = 478
        self.trans = transforms.Compose([
            transforms.Resize((256, 256), antialias=True),
        ])

    def to(self, device):
        self.model.to(device)

    def __call__(self, x):
        x = self.trans(x)
        latent, o = self.model(x)
        result = {'output': o,
                  'latent': latent}
        return result


class MediaPipeLandmarker:
    def __init__(self, model_path: str, visibility: float, presence: float):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=False,
                                               output_facial_transformation_matrixes=False,
                                               num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.visibility = visibility
        self.presence = presence

    def __call__(self, image):
        landmark = self.detector.detect(image)
        return landmark

    def to_pixel(self, image, face_landmarks):
        image = np.copy(image)
        image_rows, image_cols, _ = image.shape

        face_landmarks = face_landmarks[-1]

        landmark_list = landmark_pb2.NormalizedLandmarkList()
        landmark_list.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        landmark_list = landmark_list.landmark

        idx_to_coordinates = {}
        for idx, landmark in enumerate(landmark_list):
            if ((landmark.HasField('visibility') and landmark.visibility < self.visibility) or
                    (landmark.HasField('presence') and landmark.presence < self.presence)):
                continue
            idx_to_coordinates[idx] = self.normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                           image_cols, image_rows)
        return idx_to_coordinates

    def draw(self, image, landmark, line=True, sep=False, draw_full=True):
        mapper = Mapper(bypass=draw_full)
        pixel = self.to_pixel(image, landmark.face_landmarks)
        indexes = get_478_indexes() if draw_full else get_68_indexes()
        # indexes = get_gcn_indexes()
        palette = [(255, 0, 255),
                   (255, 255, 255),
                   (0, 255, 0),
                   (0, 0, 255),
                   (255, 0, 0),
                   (0, 255, 255),
                   (255, 255, 0),
                   (127, 127, 127),
                   (0, 0, 0)]

        if sep is False and draw_full is False:
            indexes['eyebrows'] = copy.deepcopy(indexes['left_eyebrow'] + indexes['right_eyebrow'])
            indexes['eyes'] = copy.deepcopy(indexes['left_eye'] + indexes['right_eye'])
            del indexes['left_eyebrow']
            del indexes['right_eyebrow']
            del indexes['left_eye']
            del indexes['right_eye']

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for count, items in zip(range(len(indexes)), indexes.items()):
            color = palette[count]
            key, value = items
            # if key == "contour":
            #     mapper = self.Mapper(bypass=True)
            # else:
            #     mapper = self.Mapper(bypass=draw_full)
            for begin, end in value:
                image = cv2.circle(image, pixel[mapper[begin]], 2, color, 2)
                image = cv2.circle(image, pixel[mapper[end]], 2, color, 2)
                if line:
                    image = cv2.line(image, pixel[mapper[begin]], pixel[mapper[end]], color, 1)

        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def save(filename, image):
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    @staticmethod
    def normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float,
                                        image_width: int, image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                              math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px
