import vtk

import os
files = os.listdir('.')
#print(files1)
#files = ["ArteryObjAN1-0.obj", "ArteryObjAN1-2.obj"]
'''
print("running...")
def read_obj(file):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(file)
    reader.Update()
    return reader.GetOutput()

for file in files:
    if file.split(".")[1] == "obj":
        data = read_obj(file)
        #print(data)
        file = file.split(".")[0]
        writer = vtk.vtkXMLPolyDataWriter();
        writer.SetFileName(file+".vtp");
        writer.SetInputData(data);
        writer.Write();


'''
filename = 'ArteryObjAN1-11.obj'
reader = vtk.vtkOBJReader()
reader.SetFileName(filename)
reader.Update()
data = reader.GetOutput()
print(data)
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("ArteryObjAN1-11.vtp");
writer.SetInputData(data);
writer.Write()
