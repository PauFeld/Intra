#auxiliares.py
funciones auxiliares


#radiusfrom section.py
obtengo radio a partir de los poligonos de secciones transversales

#cross-sections.py
armo archivo vtp que tiene la centerline separada en ramas(interpolacion?) CAMBIAR OUTPUT A OBJ

#radius.py
calculo el radio a partir de los poligonos

#grafo.py
genero grafo a partir de la centerline y radios (si funciona el de migue elimino)

#pipe.py
extraigo centerline (network extraction)
extraigo los poligonos de seccion transversal  a partir de la centerline y malla


#puebaparser.ipynb
notebook para probar armado de grafo (migue)


#parseObj.py
crear grafo a partir de obj+radio(migue)

0- obj a vtk

1- saco la centerline con PIPE

1.1- saco los ultimos puntos con endpoints

2- saco las secciones transversales con PIPE
3- calculo los radios con radiusfromsection
4- saco obj con centerline en ramas de cross-sections
5- armo grafo con pruebaparser


129-17 no extrae centerline
134-0
174-6
188-12
194-0
195-17
196-11
208-16
209-18
219-14
23-11
26-18
28-0


no extrae cross sections
1-5 punta cerrada
1-16 la centerline esta fea, con advancement ratio 1.1 queda mejor


1-10
1-11
1-14