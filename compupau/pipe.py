from vmtk import pypes
from vmtk import vmtkscripts
import os


#iterativo para hacerlo en todos los de la carpeta
'''
files = os.listdir('.')
for file in files:
    #print(file.split(".")[1])
    if file.split(".")[1] == "vtp":
        script = 'vmtknetworkextraction '
        input_file =  '-ifile ' + file 
        output_file = ' -ofile ' + file.split(".")[0] + 'network.vtp'
        myPype = pypes.PypeRun(script+input_file+output_file)
'''

#myPype = pypes.PypeRun("vmtknetworkextraction -ifile ArteryObjAN1-2.vtp -advancementratio 1  -ofile An.vtp")
#myPype = pypes.PypeRun("vmtkcenterlinesections -ifile ArteryObjAN1-7.vtp -centerlinesfile  ArteryObjAN1-7network.vtp -ofile ArteryObjAN1-7sections.vtp")

#1- sacar centerline
myPype = pypes.PypeRun("vmtknetworkextraction -ifile ArteryObjAN1-2.vtp -advancementratio 1.0  -ofile An.vtp")


#2- cross section
myPype = pypes.PypeRun("vmtkcenterlinesections -ifile ArteryObjAN1-0.vtp -centerlinesfile Aresampled.vtp -ofile AA.vtp")

