import cv2 as cv
import os # Crear carpetas a travez de python
import numpy as np # Matrices
import imutils

modelo='FotoDave' # creando la primer carpeta
ruta1='D:/Documentos/CURSOS/PYTHON/Proyecto Python RF/Curso/reconocimientofacial1/Data'
rutacompleta= ruta1+ '/' + modelo
if not os.path.exists(rutacompleta): # Preguntando si la ruta existe
    os.makedirs(rutacompleta) # Creando la ruta


camara=cv.VideoCapture(0)
ruidos=cv.CascadeClassifier('D:/Documentos/CURSOS/PYTHON/Proyecto Python RF/Curso/entranamientos opencv ruidos/opencv-master/data/haarcascades/haarcascade_frontalface_default.xml') #Clasificador de cascada.
id=350 #Id de cada imagen procesada
# Iniciamos con el reconocimiento de la camara
while True: # Bucle para saber si reconoce el valor o no
    respuesta,captura=camara.read() #Aqui la camara ya se esta capturando
    if respuesta==False:break # Si arroja un valor falso, que se detenga todo el codigo
    caputura=imutils.resize(captura,width=640) #colocando la resolucion de la imagen para que no sea muy pesada

    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY) # Pasando a escala de grises
    cara=ruidos.detectMultiScale(grises,1.3,5) # Aqui estaremos identificando las caras de los objetos por 
    idcaptura=captura.copy() #Se caputuran los puntos de cada imagen, para que no trabaje tanto tu computadora


    #Bluce para hacer un recorrido de cada pixel en la foto "El cuadro de la foto"
    for(x,y,e1,e2) in cara: # x & y es Izquierda y derecha. e1 & e2 es arriba y abajo. 
        # Aqui estamos dibujando el rectangulo
        cv.rectangle(captura, (x,y), (x+e1,y+e2), (0,225,0),2)
        rostrocapturado=idcaptura[y:y+e2, x:x+e1] # Este apartado sacara fragmentos de nuestro rostro para almacenarlos en carpeta
        rostrocapturado=cv.resize(rostrocapturado, (160,160),interpolation=cv.INTER_CUBIC)# Tamaño del contorno, ya sea rectangulo o cuadrado
        cv.imwrite(rutacompleta+'/imagen_{}.jpg'.format(id), rostrocapturado) # Creando la imganen y darle nombre y formato
        id=id+1
    # Mostrar al usuario
    cv.imshow("Resultado rostro",captura)
    # Darle un escape "Quitar"
    if id==500: # Decirle que se detenga a las 350 img
        break
# Para destruir las camaras
camara.release()
cv.destroyAllWindows()
