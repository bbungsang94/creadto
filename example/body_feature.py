from body_matrix import load, infer, process, measure, draw, export
from body_matrix import score
from torchvision.transforms.functional import to_pil_image


def run():
    import numpy as np
    from creadto.models.det import TokenPoseLandmarker
    video_path = r".\example\sample_video\Psick-univ-full-vert.mp4"
    video_rotate = 0
    device = "cuda"
    font_path = r".\example\fonts\Arial.ttf"

    class FakeTransform:
        def __init__(self):
            self.last = None

        def __call__(self, x, *args, **kwargs):
            self.last = np.array(x, *args, **kwargs)
            return self

        def to(self, *args, **kwargs):
            return self.last

    kp_model, kp_transform = TokenPoseLandmarker(), FakeTransform()
    sg_model, sg_transform = load.segment_model(device)

    video, frame_counts, fps, sample_frame = load.video(
        video_path=video_path,
        rotate_angle=video_rotate,
        frame_position=1
    )

    print(sample_frame)

    measure_frames = []
    measures = []

    for index, vid_frame in enumerate(video):
        if index > 60:
            break
        from PIL import Image
        # frame = to_pil_image(vid_frame)
        frame = Image.open(r"D:\dump\taes2.jpg")
        frame = frame.rotate(video_rotate, expand=True)
        height, leg, hip, shoulder, markers, _ = measure.find_real_measures(
            image_frame=frame,
            device=device,
            keypoints_model=kp_model,
            keypoints_transform=kp_transform,
            segment_model=sg_model,
            segment_transform=sg_transform
        )

        visualized_frame = draw.visualize_measures(
            height, leg, hip, shoulder, markers,
            frame, font_path=font_path
        )
        visualized_frame.save('./test2.jpg')
        measure_frames.append(visualized_frame)
        measures.append(height)

    mean, median, minim, maxim = score.best_scores(
        measures,
        0,
        200
    )

    best_score, frame_index = score.find_nearest(
        measures,
        median
    )

    #### Export Instagram Video with Measures
    export.generate_instagram_vid(
        vid_name="instameasures_hoangdo.mp4",
        vid_width=sample_frame.width,
        vid_height=sample_frame.height,
        pil_images=measure_frames,
        stop_index=frame_index,
        fps=fps,
        repeat_rate=2,
        slow_motion_rate=1
    )
