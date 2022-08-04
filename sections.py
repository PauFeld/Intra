import numpy as np
import vtk
from auxiliares import get_points_by_line, read_vtk
from collections import Counter
from vedo import *
import traceback
import os


##Load mesh and centerline
'''
#filename = 'ArteryObjAN1-0'
#msh = load(filename+".obj")
#msh.cmap('viridis', msh.points()[:,1])
#msh.show()
#centerline = read_vtk("centerlines/" + filename + "-network.vtp")
#centerline_np = get_points_by_line(centerline)

##find centerline repeated points
#splited = np.split(centerline_np, np.where(np.diff(centerline_np[:,3]))[0]+1)
e = {}# to save every branch endpoint
sum = 0
for i in range(len(splited)):
    rama = splited[i]
    start = rama[0, :3]
    e[sum] = tuple(start) #key is the point index, value coordinates
    finish = rama[rama.shape[0]-1, :3]
    sum += rama.shape[0]
    e[sum-1] = tuple(finish)
    
##keep only the repeated endpoints
b = np.array([key for key,  value in Counter(e.values()).items() if value > 1])


##list with the indexes of the repeated points
key_list = []
for element in b: #coordintaes of each repeated point
    element = tuple(element)
    for key,value in e.copy().items():
        if element == value:#if the endpoint is on the repeated list I save the index
            key_list.append(key)#key_list tiene los indices de los puntos repetidos
  
'''
##keep the largest region when the cutting plane cuts the mesh more than once
def get_largest_region(cutplane, point):
    connectivityFilter = vtk.vtkConnectivityFilter()
    connectivityFilter.SetExtractionModeToClosestPointRegion();
    #connectivityFilter.SetExtractionModeToLargestRegion();
    connectivityFilter.SetInputData(cutplane._data);
    connectivityFilter.SetClosestPoint(point);
    connectivityFilter.Update();
    #print("number of areas: ", connectivityFilter.GetNumberOfExtractedRegions())
    m = Mesh(connectivityFilter.GetOutput())

    pr = vtk.vtkProperty()

    pr.DeepCopy(cutplane.property)
    m.SetProperty(pr)
    m.property = pr
    # assign the same transformation
    m.SetOrigin(cutplane.GetOrigin())
    m.SetScale(cutplane.GetScale())
    m.SetOrientation(cutplane.GetOrientation())
    m.SetPosition(cutplane.GetPosition())
    vis = cutplane._mapper.GetScalarVisibility()
    m.mapper().SetScalarVisibility(vis)
    return m

##Functions that cuts de mesh iteratively to calculate radius
def calculate_radius (centerline, msh, key_list):
    #area = 4.0
    points_Acum = 0 #because I move between branches I need to save the point indexes from the start of the centerline
    
    c = Mesh()#where I will save all the cross section polygons
    radius_array = []#where I will save the radius value at each point

    for j in range(centerline.GetNumberOfCells()):#calculate the radius by branch to avoid problems at the connections between branches
      
        numberOfCellPoints = centerline.GetCell(j).GetNumberOfPoints();# number of points of the branch

        for i in range (numberOfCellPoints):
            
            tangent = np.zeros((3))

            weightSum = 0.0;
            ##tangent line with the previous point (not calculated at the first point)
            if (i>0):
                point0 = centerline.GetPoint(points_Acum-1);
                point1 = centerline.GetPoint(points_Acum);

                distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(point0,point1));
                
                ##vector between the two points divided by the distance
            
                tangent[0] += (point1[0] - point0[0]) / distance;
                tangent[1] += (point1[1] - point0[1]) / distance;
                tangent[2] += (point1[2] - point0[2]) / distance;
                weightSum += 1.0;


            ##tangent line with the next point (not calculated at the last one)
            if (i<numberOfCellPoints-1):
                
                point0 = centerline.GetPoint(points_Acum);
                point1 = centerline.GetPoint(points_Acum+1);

                distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(point0,point1));
                tangent[0] += (point1[0] - point0[0]) / distance;
                tangent[1] += (point1[1] - point0[1]) / distance;
                tangent[2] += (point1[2] - point0[2]) / distance;
                weightSum += 1.0;
            

            tangent[0] /= weightSum;
            tangent[1] /= weightSum;
            tangent[2] /= weightSum;


            ##cut the plane with the mesh
            #plane = Grid(pos = centerline.GetPoint(points_Acum),sx = 5, sy = 5, res = (150,150), normal = tangent)
            cutplane = Grid(pos = centerline.GetPoint(points_Acum),s = (5,5), res = (150,150), normal = tangent).cutWithMesh(msh).triangulate()
            
            
            ##create mesh with the polygon 
            vert = cutplane.points()
            faces = cutplane.faces()
            m = Mesh([vert, faces])
            
            ##calculate area and radius of the polygon obtained
            #area = cutplane.area()

            m = get_largest_region(m, centerline.GetPoint(points_Acum)) #keep only one region if it cuts the mesh multiple times
            vert = m.points()
            faces = m.faces()
            area = m.area()
            ceradius = np.sqrt(area / np.pi)
            radius_array.append(ceradius)
            #if points_Acum in key_list:
            #    print(points_Acum, centerline.GetPoint(points_Acum), ceradius) #print the repeated points
            m = Mesh([vert, faces])
            c = c + m
            points_Acum += 1 #keep track of points relative to the whole centerline not just the specific branch
           
    ##return the array with all the radius values and assembley containig the cross section polygons
    return np.array(radius_array), c
  


def save_r_p (centerline, msh, file):
    centerline_np = get_points_by_line(centerline)
    ##find centerline repeated points
    splited = np.split(centerline_np, np.where(np.diff(centerline_np[:,3]))[0]+1)
    e = {}# to save every branch endpoint
    sum = 0
    for i in range(len(splited)):
        rama = splited[i]
        start = rama[0, :3]
        e[sum] = tuple(start) #key is the point index, value coordinates
        finish = rama[rama.shape[0]-1, :3]
        sum += rama.shape[0]
        e[sum-1] = tuple(finish)
        
    ##keep only the repeated endpoints
    b = np.array([key for key,  value in Counter(e.values()).items() if value > 1])


    ##list with the indexes of the repeated points
    key_list = []
    for element in b: #coordintaes of each repeated point
        element = tuple(element)
        for key,value in e.copy().items():
            if element == value:#if the endpoint is on the repeated list I save the index
                key_list.append(key)#key_list tiene los indices de los puntos repetidos

    k = {}
    ##dictionary with the indexes and coordinates of the repeated points
    for key in key_list:
        k[key] = tuple(centerline_np[key,:3])

    ## join the points with the same coordinates, key are the coordinates and values list with the indexes
    res = {}
    for i, v in k.items():
        res[v] = [i] if v not in res.keys() else res[v] + [i]
        #print("res: ", res)

   
    #ind = []
    
    radius_array, po = calculate_radius(centerline, msh, key_list)
    for point in res:
        ra = radius_array[res[point]]
        min = np.min(ra)
        min_i = list(radius_array).index(min)
        for index in res[point]:
            #print(index)
            if radius_array[index] == min:
                #ind.append(index)
                pass
            else:
                radius_array[index] = min
                po.GetParts().ReplaceItem(index+1, po.GetParts().GetItemAsObject(min_i+1))
                #print("removed: ", index+1, min_i)


    #f = Plotter()
    #f.add(centerline)
    #f.show(po, msh.alpha(0.5))

    '''
    me = Mesh()
    for i in range(len(po.actors)):
        me = merge(me, po.actors[i])
        #print(po.actors[i])

    #print(me)
    #print(po.GetParts())

    f = Plotter()
    f.add(centerline)
    f.show(me, msh.alpha(0.5))
    '''
    np.save("radius/" + file.split(".")[0] + '-radius.npy', radius_array)
    #write(polys, "crossSections/" + filename + "-section.obj")
    #write(me, "crossSections/" + filename + "section.obj")


    ##save the assembley with the cross sections as obj file
    # Create a rendering window and renderer
    ren = vtk.vtkRenderer()
    ren.AddActor(po)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    # Create a renderwindowinteractor
    #iren = vtk.vtkRenderWindowInteractor()
    #iren.SetRenderWindow(renWin)

    # Enable user interface interactor
    #iren.Initialize()
    #renWin.Render()
    #iren.Start()

    writer = vtk.vtkOBJExporter()
    writer.SetFilePrefix("crossSections2/" + file.split(".")[0] + "-sections");
    writer.SetInput(renWin);
    writer.Write()
    return


files = os.listdir('.')
#files = ["ArteryObjAN1-7.obj"]
cfolder = os.listdir('./crossSections2')
for file in files:
    ##Load mesh and centerline
    msh = load(file)
    centerline = read_vtk("centerlines/" + file.split(".")[0] + "-network.vtp")

    if file.split(".")[1] == "obj":
        if file.split(".")[0] + '-sections.obj' not in cfolder:
            try:
                save_r_p(centerline, msh, file)
                print("Calculated: ", file)
            except Exception:
                traceback.print_exc()
                print("problem with: ", file)
 
'''
##PRUEBAS
file  = "ArteryObjAN1-7"
msh = load(file+".obj")
msh.cmap('viridis', msh.points()[:,1])
#msh.show()
centerline = read_vtk("centerlines/" + file + "-network.vtp")
#centerline_np = get_points_by_line(centerline)
tangent = np.zeros((3))
point0 = centerline.GetPoint(10);
point1 = centerline.GetPoint(11);
point2 = centerline.GetPoint(12);

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
plane = Grid( point1, s = (3, 3), res = (150,150),normal = tangent).wireframe(False)
f = Plotter()
f.add(plane)
f.add(msh)
f.show()


cutplane = plane.cutWithMesh(msh).triangulate()
f = Plotter()
f.add(cutplane)
f.add(msh)
area = cutplane.area()
#print(area)
f = Plotter()
f.show([cutplane, f"area: {area}"], axes=1)


#mesh2 = cutplane.extractLargestRegion()
#f = Plotter()
#f.show(cutplane)

    
m =get_largest_region(cutplane, point1)
f = Plotter()
f.show(m)'''