from vmtk import pypes
from vmtk import vmtkscripts
from vtk import vtkXMLPolyDataReader, vtkPolyDataReader, vtkIdList, vtkFloatArray
from vtk.util import numpy_support
import numpy as np
import math
import vtk
import os
files = os.listdir('.')



def read_vtk(filename):
    
	if filename.endswith('xml') or filename.endswith('vtp'):
		polydata_reader = vtkXMLPolyDataReader()
	else:
		polydata_reader = vtkPolyDataReader()

	polydata_reader.SetFileName(filename)
	polydata_reader.Update()

	polydata = polydata_reader.GetOutput()

	return polydata

sections = read_vtk('AA.vtp')

points_array = []
for i in range(sections.GetNumberOfPoints()):
	point = sections.GetPoint(i)
	points_array.append(point)
	#print(point)
	
points_array = np.array((points_array))

##1.1 extraigo los puntos por cell
points_array = []
number_of_sections = sections.GetNumberOfCells()
print(number_of_sections)

points_array = np.array((points_array))
print(sections.GetCell(0))
print(sections.GetCell(1))
radius_array = []
for i in range(number_of_sections):
    normals = np.empty(3)
    crossSectionArea = sections.GetCell(i).ComputeArea(sections.GetCell(i).GetPoints(), sections.GetCell(i).GetNumberOfPoints(), sections.GetCell(i).GetPointIds(), normals)
    #print(crossSectionArea)
    ceradius = np.sqrt(crossSectionArea / np.pi)
    print(i, ceradius)
    radius_array.append(ceradius)

radius_array = np.array((radius_array))
print(radius_array)

np.save('ArteryObjAN1-0radius.npy', radius_array)

crossSectionProperties = vtk.vtkMassProperties()
crossSectionProperties.SetInputData(sections.GetCell(0))
currentSurfaceArea = crossSectionProperties.GetSurfaceArea()