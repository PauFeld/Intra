from vtk import vtkXMLPolyDataReader, vtkPolyDataReader
import numpy as np
import math
import vtk


##vtk
def read_vtk(filename):
    
	if filename.endswith('xml') or filename.endswith('vtp'):
		polydata_reader = vtkXMLPolyDataReader()
	else:
		polydata_reader = vtkPolyDataReader()

	polydata_reader.SetFileName(filename)
	polydata_reader.Update()

	polydata = polydata_reader.GetOutput()

	return polydata

def get_points_by_line(centerline):
    points_array = []
    for i in range(centerline.GetNumberOfCells()):
        cell = centerline.GetCell(i)
        points = cell.GetPoints()
        for j in range(points.GetNumberOfPoints()):
            point = points.GetPoint(j)#i me dice el numero de linea y j el de punto
            p = (point[0], point[1], point[2], i)
            points_array.append(p)
    return np.array(points_array)

##interpolacion
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
    segment_length = total_length / (target_count);
    
    result = []
    
    src_segment_offset = 0
    start = 0
    finish = start + 1
    
    src_segment_length = Distance(line[start], line[finish]);

    for i in range (0, target_count):
        # calculo la longitud total del recorrido desde el inicio hasta el punto i 
        next_offset = segment_length * i;###i es numero de punto en el que estoy, next_offstet es la distancia desde el primero
        # ¿La longitud del segmento i,i+1 sumado al acumulado de la centerline sin samplear es menor a total del recorrido?
        while((src_segment_offset + src_segment_length < next_offset) and (finish <= line.shape[0]-2)):
            #  Si es menor, entonces todavía no llegue a donde tengo que poner el punto, sumo lka distancia de un punto mas en la centerline sin resampleo
            src_segment_offset += src_segment_length ### 
            start = finish
            finish = start+1
            src_segment_length = Distance(line[start], line[finish])
        
        #print(start,finish, line[start], line[finish])
        print(start, finish)
        # Si cortó el while, estoy listo para agregar un nuevo punto al resampling, excepto que me haya salido del segmento
        if (line[start][3] != line[finish][3]):
          start = finish
          finish = start+1          
        else: # si sigo en el mismo segmento, entonces calculo la posición y agrego el punto
            #print("agrega punto")
            part_offset = next_offset - src_segment_offset;
            part_ratio  = part_offset / src_segment_length;
            print("agrego")
            result.append((line[start][0] + part_ratio * (line[finish][0] - line[start][0]),
                           line[start][1] + part_ratio * (line[finish][1] - line[start][1]),
                           line[start][2] + part_ratio * (line[finish][2] - line[start][2]), 
                           line[start][3]))
        
    return result