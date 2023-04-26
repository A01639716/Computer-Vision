import numpy as np
import cv2 as cv


def metodo_watersheed():
    #Parte 1 para cargar las imagenes y hacer la escala de grises 
    images = ['road127.png','road126.png','road125.png','road124.png','road123.png','road122.png','road121.png','road120.png','road119.png','road12.png']

    for i in images:
        img = cv.imread(i)
        # Convertimos la imagen a escala de grises
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        # Aplicamos un filtro de suavizado para eliminar ruido
        blur = cv.medianBlur(gray,5)

        # Binarizamos la imagen para obtener una imagen en blanco y negro
        ret,thresh = cv.threshold(blur,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

        # Aplicamos la transformación morfológica de cierre para unir regiones
        kernel = np.ones((3,3),np.uint8)
        closing = cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel,iterations = 2)

        # Encontramos los contornos de la imagen
        contours, hierarchy = cv.findContours(closing,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

        # Creamos una matriz de marcadores y los inicializamos con ceros
        markers = np.zeros((img.shape[0], img.shape[1]),dtype=np.int32)

        # Dibujamos los contornos encontrados en los marcadores
        for i in range(len(contours)):
            cv.drawContours(markers, contours, i, (i+1), -1)

        # Agregamos un valor constante a los marcadores para separar los objetos
        markers = markers + 1

        # Aplicamos el método Watershed para segmentar la imagen
        markers = cv.watershed(img, markers)

        # Coloreamos los marcadores para visualizar las regiones segmentadas
        img[markers == -1] = [255,0,0]

        # Mostramos la imagen original y la segmentada
        cv.imshow("Original",img)
        cv.waitKey(0)

        cv.imshow("Segmentada",markers)

        # Esperamos a que el usuario presione una tecla para cerrar las ventanas
        cv.waitKey(0)
        cv.destroyAllWindows()

metodo_watersheed()
