from vmtk import pypes
from vmtk import vmtkscripts

import os
files = os.listdir('.')

'''
for file in files:
    #print(file.split(".")[1])
    if file.split(".")[1] == "vtp":
        script = 'vmtknetworkextraction '
        input_file =  '-ifile ' + file 
        output_file = ' -ofile ' + file.split(".")[0] + 'network.vtp'
        myPype = pypes.PypeRun(script+input_file+output_file)
'''
file = "ArteryObjAN1-0.vtp"
script = 'vmtknetworkeditor '
input_file =  '-ifile ' + file 
output_file = ' -ofile ' + 'AAA.'
#myPype = pypes.PypeRun("vmtknetworkextraction -ifile ArteryObjAN1-2.vtp -advancementratio 1  -ofile An.vtp")
myPype = pypes.PypeRun("vmtkcenterlinesections -ifile ArteryObjAN1-2.vtp -centerlinesfile   An.vtp -ofile AA.vtp")
#myPype = pypes.PypeRun("vmtkdistancetocenterlines -ifile ArteryObjAN1-0.vtp -centerlinesfile ArteryObjAN1-0network.vtp -centerlineradius 1 -ofile AAA.vtp")
#myPype = pypes.PypeRun("vmtkcenterlines -ifile ArteryObjAN1-0.vtp  -ofile A.vtp")
from vmtk import vtkvmtk

centerlineFilter = vtkvmtk.vtkvmtkPolyDataCenterlines()
#centerlineFilter.SetInputData(file)

'''centerlineFilter.SetSourceSeedIds(sourceId)
centerlineFilter.SetTargetSeedIds(targetId)
centerlineFilter.SetRadiusArrayName('MaximumInscribedSphereRadius')
    centerlineFilter.SetCostFunction('1/R')
    centerlineFilter.SetFlipNormals(0)
    centerlineFilter.SetAppendEndPointsToCenterlines(0)
    centerlineFilter.SetSimplifyVoronoi(0)
    centerlineFilter.SetCenterlineResampling(0)
    centerlineFilter.SetResamplingStepLength(1.0)
    centerlineFilter.Update()
    end = time.time()
    print(end-start)
    Centerlines = centerlineFilter.GetOutput()
    centerLinePoints = Centerlines.GetPoints()
    voronoiDiagram = centerlineFilter.GetVoronoiDiagram()

    voronoiPoints = vtk_to_numpy(voronoiDiagram.GetPoints().GetData())
    voronoiRadius = vtk_to_numpy(voronoiDiagram.GetPointData().GetArray('MaximumInscribedSphereRadius'))

    lines = Centerlines.GetLines()
    numberOfLines = lines.GetNumberOfCells()'''

#1- sacar centerline
#myPype = pypes.PypeRun("vmtknetworkextraction -ifile ArteryObjAN1-2.vtp -advancementratio 1.0  -ofile An.vtp")

#2- dividir en ramas
#myPype = pypes.PypeRun("vmtkbranchextractor -ifile Aresampled.vtp  -ofile ArteryObjAN1-0-branches.vtp")

#3- cross section
#myPype = pypes.PypeRun("vmtkcenterlinesections -ifile ArteryObjAN1-0.vtp -centerlinesfile Aresampled.vtp -ofile AA.vtp")

#4-smoooothing
#myPype = pypes.PypeRun("vmtkcenterlinesmoothing -ifile ArteryObjAN1-0network.vtp  -ofile AA.vtp")


from vtk import vtkXMLPolyDataReader, vtkPolyDataReader, vtkIdList, vtkFloatArray
def read_vtk(filename):
    
	if filename.endswith('xml') or filename.endswith('vtp'):
		polydata_reader = vtkXMLPolyDataReader()
	else:
		polydata_reader = vtkPolyDataReader()

	polydata_reader.SetFileName(filename)
	polydata_reader.Update()

	polydata = polydata_reader.GetOutput()

	return polydata


#distance = read_vtk('AAA.vtp.vtp')
#print(distance)