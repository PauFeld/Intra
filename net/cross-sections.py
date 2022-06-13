from vmtk import pypes
from vmtk import vmtkscripts
from vtk import vtkXMLPolyDataReader, vtkPolyDataReader, vtkIdList, vtkFloatArray
import numpy as np
import math

import os
files = os.listdir('.')


file = "ArteryObjAN1-0.vtp"
script = 'vmtknetworkeditor '

#myPype = pypes.PypeRun("vmtkcenterlinesections -ifile ArteryObjAN1-0.vtp -centerlinesfile ArteryObjAN1-0network.vtp -ofile ArteryObjAN1-0.vtpsections.vtp")

def read_vtk(filename):
    
	if filename.endswith('xml') or filename.endswith('vtp'):
		polydata_reader = vtkXMLPolyDataReader()
	else:
		polydata_reader = vtkPolyDataReader()

	polydata_reader.SetFileName(filename)
	polydata_reader.Update()

	polydata = polydata_reader.GetOutput()

	return polydata

filename = 'ArteryObjAN1-0.vtpsections'
sections = read_vtk(filename+'.vtp')
centerline = read_vtk('ArteryObjAN1-0network.vtp')

sections_lines = sections.GetLines()
n_ptsSection = sections_lines.GetSize()  # nb of points
#n_ptsLumen = lumen_lines.GetSize()  # nb of points

#print(centerline)
#print(sections.GetCell(64))


#print(sections.GetCell(64).GetPointIds().GetNumberOfIds())
#print(sections.GetCell(64).GetPointId(0))

#print(sections.GetPoint(650))
#print(sections.GetPoint(1700))
'''
print(lumen_lines)
print(sections_lines)
print(n_ptsLumen)
print(n_ptsSection)'''
