import numpy as np
import math
import vtk
from auxiliares import get_points_by_line, read_vtk, LinearCurveLength, UniformLinearInterpolation


centerline = read_vtk("centerlines/ArteryObjAN1-19-network.vtp")

points_array = get_points_by_line(centerline)
print(points_array)
#separo las ramas
splited = np.split(points_array, np.where(np.diff(points_array[:,3]))[0]+1)


#endpoints = []
e = {}
sum = 0
for i in range(len(splited)):
    rama = splited[i]
    start = rama[0, :3]
    #endpoints.append(tuple(start))
    e[sum] = tuple(start)
    finish = rama[rama.shape[0]-1, :3]
    sum += rama.shape[0]
    #endpoints.append(tuple(finish))
    e[sum-1] = tuple(finish)
    

#print("endpoints", np.array(endpoints))
#print("e", e)
#a =  list(dict.fromkeys(endpoints))    
#print("not", np.array(a))

from collections import Counter
#a son las coordenadas de los endpoints
a = np.array([key for key,  value in Counter(e.values()).items() if value == 1])
print("n", a)


#b son las coordenadas de los "endpoints" en comun
b = np.array([key for key,  value in Counter(e.values()).items() if value > 1])
print("no", b)


#encuentro los indices de los endpoints y los saco del diccionario de endpoints
#en e quedan los que voy a sacar de points array
key_list = []
for element in b:
    element = tuple(element)
    for key,value in e.copy().items():
        if element == value:
            key_list.append(key)
            e.pop(key)

print("e", e)
#print("centerline", points_array)
'''
points_array = list(points_array)
count = 0
for key in e.keys():
    print(count)
    points_array.pop(key-count)
    count+=1
'''
print(list(e.values()))
points_array = np.delete(points_array, (list(e.keys())), axis=0)


#points_array = np.array((points_array))
print(points_array)


points_full = vtk.vtkPoints()
number_points = points_array.shape[0]
print(number_points)

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
       
    number_points = splited[i].shape[0]
    polyLine = vtk.vtkPolyLine()
    polyLine.GetPointIds().SetNumberOfIds(number_points-1)
    
    j = 0
    for k in range(point_count+1, point_count+number_points):
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

