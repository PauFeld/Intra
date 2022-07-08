import numpy as np
import math
import vtk
from auxiliares import get_points_by_line, read_vtk


from vedo import *
filename = 'ArteryObjAN1-17.obj'
msh = load(filename)
msh.cmap('viridis', msh.points()[:,1])
#msh.show()

centerline = read_vtk("centerlines/ArteryObjAN1-17-network.vtp")
centerline_np = get_points_by_line(centerline)



def calculate_radius (centerline):
    centerline_np = get_points_by_line(centerline)
    
    radius_array = []

    for j in range(centerline.GetNumberOfCells()):
        numberOfCellPoints = centerline.GetCell(j).GetNumberOfPoints();

        for i in range (numberOfCellPoints):

            tangent = np.zeros((3))

            weightSum = 0.0;
            if (i>0):
                point0 = centerline.GetPoint(i-1);
                point1 = centerline.GetPoint(i);

                distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(point0,point1));
                #print(distance)
                tangent[0] += (point1[0] - point0[0]) / distance;
                tangent[1] += (point1[1] - point0[1]) / distance;
                tangent[2] += (point1[2] - point0[2]) / distance;
                weightSum += 1.0;

            if (i<numberOfCellPoints-1):
            
                
                point0 = centerline.GetPoint(i);
                point1 = centerline.GetPoint(i+1);
                distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(point0,point1));
                tangent[0] += (point1[0] - point0[0]) / distance;
                tangent[1] += (point1[1] - point0[1]) / distance;
                tangent[2] += (point1[2] - point0[2]) / distance;
                weightSum += 1.0;
            

            tangent[0] /= weightSum;
            tangent[1] /= weightSum;
            tangent[2] /= weightSum;

            #plane = Grid( pos = centerline_np[i, :3],s = (3,3), res = (150,150), normal = tangent).wireframe(False)
            cutplane = Grid(pos = centerline_np[i, :3],s = (3,3), res = (150,150), normal = tangent).cutWithMesh(msh).triangulate()
            area = cutplane.area()
            ceradius = np.sqrt(area / np.pi)
            radius_array.append(ceradius)
        
    return np.array(radius_array)
  

radius_array=calculate_radius(centerline)

#np.save('radius/ArteryObjAN6-2-radius.npy', radius_array)


'''
##prueba con bunny
msh2 = Mesh(dataurl+'bunny.obj')#.scale(3).shift(0,-0.5,0.01)
plane = Grid(resx=100, resy=100).wireframe(False)
f = Plotter()
f.add(plane)
f.show(msh2)


#corto
cutplane = plane.clone().cutWithMesh(msh2).triangulate()
area = cutplane.area()
f = Plotter()
f.show([cutplane, f"area: {area}"], axes=1)
'''


#pruebo con la malla
tangent = np.zeros((3))
point0 = centerline.GetPoint(9);
point1 = centerline.GetPoint(10);
point2 = centerline.GetPoint(11);

distance1 = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(point0,point1));

distance2 = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(point1,point2));
#print(distance)
tangent[0] += (point1[0] - point0[0]) / distance1;
tangent[1] += (point1[1] - point0[1]) / distance1;
tangent[2] += (point1[2] - point0[2]) / distance1;

tangent[0] += (point2[0] - point1[0]) / distance2;
tangent[1] += (point2[1] - point1[1]) / distance2;
tangent[2] += (point2[2] - point1[2]) / distance2;

tangent[0] /= 2.0;
tangent[1] /= 2.0;
tangent[2] /= 2.0;

#m = msh.cutWithPlane(origin = centerline_np[10, :3], normal=(1,1,1))
plane = Grid( pos = centerline_np[10, :3], s = (3, 3), res = (150,150),normal = tangent).wireframe(False)
f = Plotter()
f.add(plane)
f.add(msh)
f.show()



cutp = plane.clone(deep = False).cutWithMesh(msh).triangulate()
cutplane = plane.cutWithMesh(msh).triangulate()
f = Plotter()
f.add(cutplane)
f.add(msh)
#f.show(cutp)
area = cutplane.area()
print(area)
f = Plotter()
f.show([cutplane, f"area: {area}"], axes=1)

