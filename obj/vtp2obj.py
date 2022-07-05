import vtk
from auxiliares import read_vtk

filename = 'centerlines/ArteryObjAN1-4-network.vtp'
centerline = read_vtk(filename)

writer = vtk.vtkOBJExporter()
writer.SetFilePrefix("centerlines/ArteryObjAN1-4-network.obj");
writer.SetInput(centerline);
writer.Write()
