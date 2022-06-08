from vmtk import pypes
from vmtk import vmtkscripts

import os
files = os.listdir('.')


for file in files:
    #print(file.split(".")[1])
    if file.split(".")[1] == "vtp":
        #print(file)
        #print("hy")
        script = 'vmtknetworkextraction '
        input_file =  '-ifile ' + file 
        output_file = ' -ofile ' + file.split(".")[0] + 'network.vtp'
        myPype = pypes.PypeRun(script+input_file+output_file)
