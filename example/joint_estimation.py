import facer
from creadto.services.estimation import estimate_body_joint_from_file, estimate_body_joint_from_directory

result = estimate_body_joint_from_file("./sample/m-daniel-half.jpeg")
for key, value in result.items():
    if "images" in key:
        print(key)
        value = (value - value.min()) / (value.max() - value.min())
        facer.show_bchw(value * 250.)

result = estimate_body_joint_from_directory("./sample")
for key, value in result.items():
    if "images" in key:
        print(key)
        value = (value - value.min()) / (value.max() - value.min())
        facer.show_bchw(value * 250.)
