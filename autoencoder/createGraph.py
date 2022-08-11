from parseObj import calcularMatriz
import pickle
import os
import traceback


from vedo import *
##Load mesh and centerline
#file.split(".")[0].split("-")[0]
c = 0
files = os.listdir('.')
gfolder = os.listdir('./grafos')
#files = ["ArteryObjAN1-7.obj"]
for file in files:
    ##Load mesh and centerline
    fileObj = open("centerlines/ " +file.split(".")[0] +"-network.obj")
    
    if file.split(".")[1] == "obj":
        if file.split(".")[0] + '-grafo.gpickle' not in gfolder:
            try: 
                grafo = calcularMatriz(fileObj, "radius/" + file.split(".")[0] + "-radius.npy")
                print("calculating: ", file)
            
                with open("grafos/" + file.split(".")[0] + '-grafo.gpickle', 'wb') as f:
                    pickle.dump(grafo, f, pickle.HIGHEST_PROTOCOL)
            except Exception:
                traceback.print_exc()
                print("problem: ", file)
                c += 1
                print(c)



