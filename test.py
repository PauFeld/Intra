from vmtk import pypes
from vmtk import vmtkscripts
from vtk import vtkXMLPolyDataReader, vtkPolyDataReader, vtkIdList, vtkFloatArray
from vtk.util import numpy_support
import numpy as np
import math
import vtk
import os

origin = [0.0, 0.0, 0.0]
p0 = [1.0, 0.0, 0.0]
p1 = [2.0, 0.0, 0.0]
p2 = [3.0, 0.0, 0.0]
p3 = [4.0, 0.0, 0.0]

p4 = [2.0, 1.0, 0.0]
p5 = [2.0, 2.0, 0.0]
p6 = [2.0, 3.0, 0.0]

# Create a vtkPoints object and store the points in it
points = vtk.vtkPoints()
points.InsertNextPoint(origin)
points.InsertNextPoint(p0)
points.InsertNextPoint(p1)
points.InsertNextPoint(p2)
points.InsertNextPoint(p3)

points.InsertNextPoint(p4)
points.InsertNextPoint(p5)
points.InsertNextPoint(p6)

print("hola")
polyLine = vtk.vtkPolyLine()

polyLine.GetPointIds().SetNumberOfIds(5)

for i in range(0, 5):
    polyLine.GetPointIds().SetId(i, i)

# Create a cell array to store the lines in and add the lines to it
cells = vtk.vtkCellArray()
cells.InsertNextCell(polyLine)

polyLine = vtk.vtkPolyLine()
polyLine.GetPointIds().SetNumberOfIds(3)

#for i in range(5, 8):
#    polyLine.GetPointIds().SetId(i, i)

polyLine.GetPointIds().SetId(0, 4)
polyLine.GetPointIds().SetId(1, 5)
polyLine.GetPointIds().SetId(2, 6)
cells.InsertNextCell(polyLine)

# Create a polydata to store everything in
polyData = vtk.vtkPolyData()

# Add the points to the dataset
polyData.SetPoints(points)

# Add the lines to the dataset
polyData.SetLines(cells)
print("l1", polyData.GetCell(0))
print("l2",polyData.GetCell(1))