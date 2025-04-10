
import facer
from creadto.services.reconstruction import reconstruct_head_from_directory, reconstruct_head_from_file


result = reconstruct_head_from_file("./sample/m-daniel-half.jpeg")
for key, value in result.items():
    if "images" in key:
        print(key)
        facer.show_bchw(value * 255.)

result = reconstruct_head_from_directory("./sample")
for key, value in result.items():
    if "images" in key:
        print(key)
        facer.show_bchw(value * 255.)
