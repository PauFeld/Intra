import numpy as np
import math
import vtk
from auxiliares import get_points_by_line, read_vtk, LinearCurveLength, UniformLinearInterpolation


centerline = read_vtk('ArteryObjAN1-7network.vtp')

##1-Extraigo los puntos de la centerline
#print(centerline.GetPoints())
points_array = []
for i in range(centerline.GetNumberOfPoints()):
	point =centerline.GetPoint(i)
	points_array.append(point)
	
	
points_array = np.array((points_array))

##1.1 extraigo los puntos por cell, es decir a que linea pertenece cada punto
points_array = get_points_by_line(centerline)
        
		
points_array = np.array((points_array))

for i in range(points_array.shape[0]):
    print(i, points_array[i])


interpolated = np.array(UniformLinearInterpolation(points_array, 150))


print('\n')
print( "Source linear interpolated length:           ", LinearCurveLength(points_array), '\n')
print( "Interpolation's linear interpolated length:  ", LinearCurveLength(interpolated), '\n')

print( "Original number of points:                   ", points_array.shape[0], '\n')
print( "Interpolation's number of points:            ", interpolated.shape[0], '\n')

#np.save('AResampled.npy', interpolated)


##3- creo polyline con los puntos de la centerline y polydata con las ramas
# Create a vtkPoints object and store the points in it

interpolated = points_array
points_full = vtk.vtkPoints()

number_points = interpolated.shape[0]

for i in range(number_points):
    points_full.InsertNextPoint(interpolated[i,:3])

# Create a polydata to store everything in
polyData = vtk.vtkPolyData()

# Add the points to the dataset
polyData.SetPoints(points_full)

    
splited = np.split(interpolated, np.where(np.diff(interpolated[:,3]))[0]+1)

point_count = 0
cells = vtk.vtkCellArray()
for i in range(len(splited)):
    print("point count", point_count)
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
#print("l0", polyData.GetCell(0))
#print("l1", polyData.GetCell(1))


writer = vtk.vtkXMLPolyDataWriter();
writer.SetFileName("Aresampled.vtp");
writer.SetInputData(polyData);
writer.Write();

