import numpy as np
import vtk
from auxiliares import get_points_by_line, read_vtk
from collections import Counter
from vedo import *


##Load mesh and centerline
filename = 'ArteryObjAN1-0'
data = read_vtk("crossSections/section.vtp")

print(data.GetPointData().GetArray(0))

print(data.GetPointData().GetArray(0).GetValue(12350))

