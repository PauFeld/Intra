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
#myPype = pypes.PypeRun("vmtknetworkextraction -ifile ArteryObjAN1-0.vtp -advancementratio 1.0  -ofile An.vtp")
myPype = pypes.PypeRun("vmtkcenterlinesections -ifile ArteryObjAN1-0.vtp -centerlinesfile An.vtp -ofile AA.vtp")
#myPype = pypes.PypeRun("vmtkdistancetocenterlines -ifile ArteryObjAN1-0.vtp -centerlinesfile ArteryObjAN1-0network.vtp -ofile AAA.vtp")
#myPype = pypes.PypeRun("vmtkcenterlines -ifile ArteryObjAN1-10.vtp -seedselector carotidprofiles  -ofile A1-10.vtp")
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