from vtk import vtkXMLPolyDataReader, vtkPolyDataReader
import numpy as np
import math
import vtk
from auxiliares import read_vtk, get_points_by_line


centerline = read_vtk('ArteryObjAN1-7network.vtp')


##1 extraigo los puntos por cell, es decir a que linea pertenece cada punto
points_array = get_points_by_line(centerline)

print(points_array.shape)

number_points = points_array.shape[0]

points_full = vtk.vtkPoints()
for i in range(number_points):
    points_full.InsertNextPoint(points_array[i,:3])

# Create a polydata to store everything in
polyData = vtk.vtkPolyData()

# Add the points to the dataset
polyData.SetPoints(points_full)

    
splited = np.split(points_array, np.where(np.diff(points_array[:,3]))[0]+1)

point_count = 0
cells = vtk.vtkCellArray()
for i in range(len(splited)):
    #splited[i] es la rama i
    number_points = splited[i].shape[0]
    polyLine = vtk.vtkPolyLine()
    polyLine.GetPointIds().SetNumberOfIds(number_points)
    
    j = 0
    for k in range(point_count, point_count+number_points):
        polyLine.GetPointIds().SetId(j, k)
        j += 1
    point_count+=number_points
    cells.InsertNextCell(polyLine)


polyData.SetLines(cells)

#writer = vtk.vtkXMLPolyDataWriter();
#writer.SetFileName("Aresampled.vtp");
#writer.SetInputData(polyData);
#writer.Write();


sections = read_vtk('ArteryObjAN1-7sections.vtp')

number_of_sections = sections.GetNumberOfCells()

radius_array = []
for i in range(number_of_sections):
    normals = np.empty(3)
    crossSectionArea = sections.GetCell(i).ComputeArea(sections.GetCell(i).GetPoints(), sections.GetCell(i).GetNumberOfPoints(), sections.GetCell(i).GetPointIds(), normals)
    ceradius = np.sqrt(crossSectionArea / np.pi)
    radius_array.append(ceradius)

radius_array = np.array((radius_array)).reshape(63, 1)
print(radius_array.shape)

#np.save('ArteryObjAN1-0radius.npy', radius_array)

points_with_radius = np.concatenate((points_array, radius_array), axis = 1)
print(points_with_radius)

np.save("ArteryObjAN1-7", points_with_radius)