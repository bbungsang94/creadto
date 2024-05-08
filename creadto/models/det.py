import copy
import math
from typing import Union, Tuple

import cv2
import torch
import numpy as np
from torchvision.transforms import transforms


class TokenPoseLandmarker:
    def __init__(self, device="cuda:0"):
        from easydict import EasyDict
        from mmpose.apis import process_mmdet_results, inference_top_down_pose_model
        from mmdet.apis import inference_detector, init_detector
        from creadto._external.PCT.demo_img_with_mmdet import init_pose_model, vis_pose_result
        from mmpose.datasets import DatasetInfo

        args = {
            'det_config': "./creadto-model/cascade_rcnn_x101_64x4d_fpn_coco.py",
            'det_checkpoint': "./creadto-model/cascade_rcnn_x101_64x4d_fpn_20e_coco_20200509_224357-051557b1.pth",
            'pose_config': "./creadto-model/pct_large_classifier.py",
            'pose_checkpoint': "./creadto-model/swin_large.pth",
            'det_cat_id': 1,
            'thickness': 2,
            'bbox_thr': 0.3,
            'device': device
        }
        args = EasyDict(args)
        det_model = init_detector(args.det_config, args.det_checkpoint, device=args.device.lower())
        pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=args.device.lower())
        dataset = pose_model.cfg.data['test']['type']
        dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
        dataset_info = DatasetInfo(dataset_info)

        self.models = {
            'det': det_model,
            'pose': pose_model
        }
        self.func = {
            'det': inference_detector,
            'proc': process_mmdet_results,
            'top_down': inference_top_down_pose_model,
            'vis': vis_pose_result
        }
        self.utils = {
            'dataset': dataset,
            'dataset_info': dataset_info,
        }
        self.parameters = {
            'det_cat_id': args.det_cat_id,
            'bbox_thr': args.bbox_thr
        }

    def __call__(self, images=None, filenames=None):
        if images is None:
            images = filenames
        result = []
        for image in images:
            # test a single image, the resulting box is (x1, y1, x2, y2)
            det_boxes = self.func['det'](self.models['det'], image)
            # keep the person class bounding boxes.
            det_result = self.func['proc'](det_boxes, self.parameters['det_cat_id'])
            pose_result, info = self.func['top_down'](
                self.models['pose'],
                image,
                det_result,
                bbox_thr=self.parameters['bbox_thr'],
                format='xyxy',
                dataset=self.utils['dataset'],
                dataset_info=self.utils['dataset_info'],
                return_heatmap=False,
                outputs=None)
            stub = dict()
            stub['keypoints'] = torch.stack([torch.tensor(x['keypoints'], dtype=torch.float32) for x in pose_result])
            stub['scores'] = torch.stack([torch.tensor(x['bbox'][-1], dtype=torch.float32) for x in pose_result])
            stub['boxes'] = torch.stack([torch.tensor(x['bbox'][:4], dtype=torch.float32) for x in pose_result])
            center_x = (stub['boxes'][:, 2] + stub['boxes'][:, 0]) / 2.
            center_y = (stub['boxes'][:, 3] + stub['boxes'][:, 1]) / 2.
            stub['center'] = torch.cat([center_x, center_y])
            stub['info'] = info
            result.append(stub)
        return result


class GenderClassification:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.label = ['female', 'male']
        self.encoder = AutoImageProcessor.from_pretrained("rizvandwiki/gender-classification")
        self.model = AutoModelForImageClassification.from_pretrained("rizvandwiki/gender-classification")

    def __call__(self, x):
        o = self.model(x)
        output = [self.label[torch.argmax(i)] for i in o.logits]
        return output


class FaceAlignmentLandmarker:
    def __init__(self):
        import face_alignment

        self.scale_factor = 1.25
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    def to_point(self, left, right, top, bottom):
        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0,
                           bottom - (bottom - top) / 2.0])

        size = int(old_size * self.scale_factor)
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        return src_pts

    def __call__(self, image):
        """
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        """
        output = {'bbox': None, 'points': None, 'type': 'kpt68'}
        out = self.model.get_landmarks(image)
        if out is not None:
            kpt = out[0].squeeze()
            left = np.min(kpt[:, 0])
            right = np.max(kpt[:, 0])
            top = np.min(kpt[:, 1])
            bottom = np.max(kpt[:, 1])
            bbox = [left, top, right, bottom]

            output['points'] = self.to_point(left, right, top, bottom)
            output['bbox'] = bbox

        return output


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
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=False,
                                               output_facial_transformation_matrixes=False,
                                               num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.visibility = visibility
        self.presence = presence

    def __call__(self, image):
        # import mediapipe as mp
        # mp.Image.
        landmark = self.detector.detect(image)
        return landmark

    def to_pixel(self, image, face_landmarks):
        from mediapipe.framework.formats import landmark_pb2

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
        from creadto._external.mediapipe.convention import get_478_indexes, get_68_indexes, Mapper

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
