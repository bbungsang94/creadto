
import facer
from creadto.services.detection import detect_face_from_directory, detect_face_from_file


crop_image = detect_face_from_file("./sample/m-daniel-half.jpeg")
facer.show_bchw(crop_image.unsqueeze(dim=0) * 255.)

crop_images = detect_face_from_directory("./sample")
facer.show_bchw(crop_images * 255.)
