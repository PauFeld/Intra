from vmtk import pypes
from vmtk import vmtkscripts
from vtk import vtkXMLPolyDataReader, vtkPolyDataReader, vtkIdList, vtkFloatArray
from vtk.util import numpy_support
import numpy as np
import math
import vtk
import os
files = os.listdir('.')


file = "ArteryObjAN1-0.vtp"
script = 'vmtknetworkeditor '

#myPype = pypes.PypeRun("vmtkcenterlinesections -ifile ArteryObjAN1-0.vtp -centerlinesfile ArteryObjAN1-0network.vtp -ofile ArteryObjAN1-0.vtpsections.vtp")

def read_vtk(filename):
    
	if filename.endswith('xml') or filename.endswith('vtp'):
		polydata_reader = vtkXMLPolyDataReader()
	else:
		polydata_reader = vtkPolyDataReader()

	polydata_reader.SetFileName(filename)
	polydata_reader.Update()

	polydata = polydata_reader.GetOutput()

	return polydata

#filename = 'ArteryObjAN1-0.vtpsections'
#sections = read_vtk(filename+'.vtp')
centerline = read_vtk('ArteryObjAN1-0network.vtp')

##1-Extraigo los puntos de la centerline
#print(centerline.GetPoints())
points_array = []
for i in range(centerline.GetNumberOfPoints()):
	point =centerline.GetPoint(i)
	points_array.append(point)
	#print(point)
	
points_array = np.array((points_array))

##1.1 extraigo los puntos por cell
points_array = []
number_of_lines = centerline.GetNumberOfCells()
for i in range(centerline.GetNumberOfCells()):
    cell = centerline.GetCell(i)
    points = cell.GetPoints()
    for j in range(points.GetNumberOfPoints()):
        point = points.GetPoint(j)#i me dice el numero de linea y j el de punto
        #print(i,j,point)
        p = (point[0], point[1], point[2], i)
        points_array.append(p)
	#
    # 	
points_array = np.array((points_array))
#print(points_array)
#print("c0", centerline.GetCell(0))
#print("c1", centerline.GetCell(1))
#print("c2", centerline.GetCell(2))
##2- Resampleo los puntos
#Calculo distancia euclidea entre dos puntos susecivos
def Distance(pointA, pointB):
    xa, ya, za, la = pointA
    xb, yb, zb, lb = pointB
    dist = math.sqrt((xa-xb)**2+(ya-yb)**2+(za-zb)**2)
    return dist;


def LinearCurveLength(line):
    accum = 0
    npoints = line.shape[0]
    for point in range(0,npoints-1):
        pointA = line[point]
        pointB= line[point+1]
        accum = accum + Distance(pointA, pointB)
    return accum
        

def UniformLinearInterpolation(line, target_count):
    
    #target_count es el numero de puntos final
    total_length = LinearCurveLength(line)
    segment_length = total_length / (target_count - 1);
    
    result = []
    
    src_segment_offset = 0
    start = 0
    finish = start + 1
    
    src_segment_length = Distance(line[start], line[finish]);
    
    i = 1
    for i in range (0, target_count):
        next_offset = segment_length * i;###i es numero de punto en el que estoy, next_offstet es la distancia desde el primero
        while(src_segment_offset + src_segment_length < next_offset-1 and finish < line.shape[0]-1):
            src_segment_offset += src_segment_length ### check
            start = finish
            finish = start+1
            src_segment_length = Distance(line[start], line[finish])

        if (line[start][3] != line[finish][3]):
          start = finish
          finish = start+1

		
        part_offset = next_offset - src_segment_offset;
        part_ratio = part_offset / src_segment_length;
       
        result.append((line[start][0] + part_ratio * (line[finish][0] - line[start][0]),
                       line[start][1] + part_ratio * (line[finish][1] - line[start][1]),
                       line[start][2] + part_ratio * (line[finish][2] - line[start][2]),line[start][3]))
        
    return result


interpolated = np.array(UniformLinearInterpolation(points_array, 100))

print("Interpolated Points:\n")
#for point in interpolated:
#    print (point)



print('\n')
print( "Source linear interpolated length:           ", LinearCurveLength(points_array), '\n')
print( "Interpolation's linear interpolated length:  ", LinearCurveLength(interpolated), '\n')

print( "Original number of points:                   ", points_array.shape[0], '\n')
print( "Interpolation's number of points:            ", interpolated.shape[0], '\n')

#np.save('AResampled.npy', interpolated)


##3- Exporto como .vtu la centerline resampleada

'''
interpolated_vtk = numpy_support.numpy_to_vtk(num_array=interpolated)
writer = vtk.vtkArrayWriter();
writer.SetFileName("Aresampled.vtk");



#writer.SetInputData(interpolated_vtk);
#writer.Write();
a = vtk.vtkPolyData()
points = vtk.vtkPoints()
print(interpolated_vtk.GetTuple(0))
#points.InsertPoint()
#a.SetPoints(points)
#print(a)
#writer.SetInputData(interpolated_vtk);
writer.Write();	
#sections_lines = sections.GetLines()
#n_ptsSection = sections_lines.GetSize()  # nb of points
#n_ptsLumen = lumen_lines.GetSize()  # nb of points

#print(centerline)
#print(sections.GetCell(64))


#print(sections.GetCell(64).GetPointIds().GetNumberOfIds())
#print(sections.GetCell(64).GetPointId(0))

#print(sections.GetPoint(650))
#print(sections.GetPoint(1700))


from pyevtk.hl import pointsToVTK

x = np.array((interpolated[:,0]))
y = np.array((interpolated[:,1]))
z = np.array((interpolated[:,2]))
#pointsToVTK("apoints", x, y, z)
'''

##4- creo polyline con los puntos de la centerline
# Create a vtkPoints object and store the points in it
points_full = vtk.vtkPoints()
#interpolated = points_array
number_points = interpolated.shape[0]
#number_points = interpolated.shape[0]
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
    #points = vtk.vtkPoints()
    
    #cells = vtk.vtkCellArray()
     
    #print(number_points)
    #print("s", splited[i])
    #for point in splited[i]:
        #print(point[0:3])
     #   points.InsertNextPoint(point[0:3])
    #polyLine.GetPointIds().SetNumberOfIds(number_points)
    j = 0
    for k in range(point_count, point_count+number_points):
        polyLine.GetPointIds().SetId(j, k)
        j += 1
    point_count+=number_points
    #print(i, polyLine)
    cells.InsertNextCell(polyLine)

#for i in range(0, number_points):
#    polyLine.GetPointIds().SetId(i, i)

polyData.SetLines(cells)
print("l0", polyData.GetCell(0))
print("l1", polyData.GetCell(1))
#print(cells.GetPointIds())
#polyLine = vtk.vtkPolyLine()
#polyLine.GetPointIds().SetNumberOfIds(number_points)
#for i in range(0, number_points):
#    polyLine.GetPointIds().SetId(i, i)
# Create a cell array to store the lines in and add the lines to it
#cells = vtk.vtkCellArray()
#cells.InsertNextCell(polyLine)



# Add the lines to the dataset



writer = vtk.vtkXMLPolyDataWriter();
writer.SetFileName("Aresampled.vtp");
writer.SetInputData(polyData);
writer.Write();
'''
print(lumen_lines)
print(sections_lines)
print(n_ptsLumen)
print(n_ptsSection)'''

