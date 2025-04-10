
import facer
from creadto.services.segmentation import segment_face_from_file, segment_face_from_directory


seg_result = segment_face_from_file("./sample/m-daniel-half.jpeg")
facer.show_bchw(seg_result['vis_image'])

seg_result = segment_face_from_directory("./sample")
facer.show_bchw(seg_result['vis_image'])
