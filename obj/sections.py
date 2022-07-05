import numpy as np
import math
from pyparsing import original_text_for
import vtk
from auxiliares import get_points_by_line, read_vtk, LinearCurveLength, UniformLinearInterpolation


from vedo import *
filename = 'ArteryObjAN1-11.vtp'
msh = load(filename)
msh.cmap('viridis', msh.points()[:,1])
#msh.show()

centerline = read_vtk("centerlines/ArteryObjAN1-11-network.vtp")
centerline_np = get_points_by_line(centerline)
numberOfCellPoints = centerline.GetNumberOfPoints();
#print(numberOfCellPoints)
radius_array = []
for i in range (numberOfCellPoints):

    point = centerline.GetPoint(i);

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

    plane = Grid( pos = centerline_np[i, :3],s = (3,3), res = (100,100), normal = tangent).wireframe(False)
    cutplane = plane.clone().cutWithMesh(msh).triangulate()
    area = cutplane.area()
    ceradius = np.sqrt(area / np.pi)
    radius_array.append(ceradius)
    #now cut branch with plane and get section. Compute section properties and store them.
  

radius_array=np.array(radius_array)
print(radius_array.shape)
    
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



#pruebo con la malla
tangent = np.zeros((3))
point0 = centerline.GetPoint(20);
point1 = centerline.GetPoint(21);

distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(point0,point1));
#print(distance)
tangent[0] += (point1[0] - point0[0]) / distance;
tangent[1] += (point1[1] - point0[1]) / distance;
tangent[2] += (point1[2] - point0[2]) / distance;

#m = msh.cutWithPlane(origin = centerline_np[10, :3], normal=(1,1,1))
plane = Grid( pos = centerline_np[20, :3],s = (3,3), res = (100,100), normal = tangent).wireframe(False)
#cutplane = msh.clone().cutWithPlane(origin = centerline_np[20, :3],normal = (1,0,1))
cutplane = plane.clone().cutWithMesh(msh).triangulate()
f = Plotter()
f.add(plane)
f.add(msh)
f.show()
area = cutplane.area()
print(area)
f = Plotter()
f.show([cutplane, f"area: {area}"], axes=1)
'''
def cutWithPlane(self, origin=(0, 0, 0), normal=(1, 0, 0)):
    
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin)
    plane.SetNormal(normal)

    clipper = vtk.vtkClipPolyData()
    clipper.SetInputData(self.polydata(True)) # must be True
    clipper.SetClipFunction(plane)
    #clipper.GenerateClippedOutputOff()
    #clipper.GenerateClipScalarsOff()
    #clipper.SetValue(0)
    clipper.Update()

    cpoly = clipper.GetOutput(0)

    cutter = vtk.vtkCutter()
    cutter.SetInputData(self.polydata(True)) # must be True
    cutter.SetClipFunction(plane)
    #clipper.GenerateClippedOutputOff()
    #clipper.GenerateClipScalarsOff()
    #clipper.SetValue(0)
    clipper.Update()

    cpoly = clipper.GetOutput(0)

    return cpoly


c = cutWithPlane(msh)
f = Plotter()
f.add(plane)
f.show(msh)
'''