import numpy as np
import open3d as o3d
from creadto.services.estimation.dimension import estimate_dimension_from_file, estimate_dimension_from_directory

   
print(estimate_dimension_from_file("./sample/m-daniel-half.jpeg"))

print(estimate_dimension_from_directory("./sample"))
