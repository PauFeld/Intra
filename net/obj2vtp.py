import vtk

import os
files1 = os.listdir('.')
#print(files1)
files = ["ArteryObjAN1-0.obj", "ArteryObjAN1-2.obj"]

print("running...")
def read_obj(file):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(file)
    reader.Update()
    return reader.GetOutput()

for file in files1:
    if file.split(".")[1] == "obj":
        data = read_obj(file)
        #print(data)
        file = file.split(".")[0]
        writer = vtk.vtkXMLPolyDataWriter();
        writer.SetFileName(file+".vtp");
        writer.SetInputData(data);
        writer.Write();


'''
filename = 'ArteryObjAN1-0.obj'
reader = vtk.vtkOBJReader()
reader.SetFileName(filename)
reader.Update()
data = reader.GetOutput()
print(data)
writer = vtk.vtkXMLPolyDataWriter()
writer.SetFileName("Artery.vtp");
writer.SetInputData(data);
writer.Write()
'''