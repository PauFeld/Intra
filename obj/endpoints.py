import numpy as np
import math
import vtk
from auxiliares import get_points_by_line, read_vtk, LinearCurveLength, UniformLinearInterpolation

a = [1, 2]
b = [3, 4]
c = [1, 2]
d = np.array((a,b,c))
for row in d:
    print(row)
print(a == c)
centerline = read_vtk("centerlines/ArteryObjAN1-11-network.vtp")

points_array = get_points_by_line(centerline)

#separo las ramas
splited = np.split(points_array, np.where(np.diff(points_array[:,3]))[0]+1)

endpoints = []
s = [0, 0, 0]
f = [0, 0, 0]

for i in range(len(splited)):
    rama = splited[i]
    start = rama[0, :3]
    print(np.all(start))
    finish = rama[rama.shape[0]-1, :3]
   
    endpoints.append(start)
    endpoints.append(finish)

end = []
import functools
for i in range(len(endpoints)):
    point = [endpoints[i]]
    for j in range(i+1, len(endpoints)-1):
        po = [endpoints[j]]
        #if functools.reduce(lambda x, y : x and y, map(lambda p, q: p == q,po,point), True): 
            #print(i, j,po, point[0], po[0])
            #print(point[0] == po[0])
            #end.append(point)
        a = point == po
        if a:
            end.append(point)
            

endpoints = np.array(endpoints)
print(np.array(end))
print("endpoints", endpoints)