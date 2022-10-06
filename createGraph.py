from parseObj import calcularMatriz
import pickle
import os
import traceback


from vedo import *
##Load mesh and centerline
#file.split(".")[0].split("-")[0]
c = 0
#files = os.listdir('.')
gfolder = os.listdir('./obj/grafos')
files = ["ArteryObjAN26-4.obj"]
for file in files:
    ##Load mesh and centerline
    fileObj = open("obj/centerlines/ " +file.split(".")[0] +"-network.obj")
    
    if file.split(".")[1] == "obj":
        
        #if file.split(".")[0] + '-grafo.gpickle' not in gfolder:
        if True:
            try: 
                grafo = calcularMatriz(fileObj, "obj/radius/" + file.split(".")[0] + "-radius.npy")
                print("calculating: ", file)
            
                with open("obj/grafos/" + file.split(".")[0] + '-grafo.gpickle', 'wb') as f:
                    pickle.dump(grafo, f, pickle.HIGHEST_PROTOCOL)
            except Exception:
                traceback.print_exc()
                print("problem: ", file)
                c += 1
                print(c)



