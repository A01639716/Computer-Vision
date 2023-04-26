import numpy as np
import cv2 

def metodo_watersheed():
    #Parte 1 para cargar las imagenes y hacer la escala de grises 
    images = ['road0.png','road1.png','road10.png','road100.png','road101.png','road102.png','road103.png','road104.png','road105.png','road106.png']

    for i in images:
        img1 = cv2.imread(i)
        img = cv2.convertScaleAbs(img1)
        # Convertimos la imagen a escala de grises
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Aplicamos un filtro de suavizado para eliminar ruido
        blur = cv2.medianBlur(gray,5)

        # Binarizamos la imagen para obtener una imagen en blanco y negro
        ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        # Aplicamos la transformación morfológica de cierre para unir regiones
        kernel = np.ones((3,3),np.uint8)
        closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel,iterations = 2)

        # Encontramos los contornos de la imagen
        contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # Creamos una matriz de marcadores y los inicializamos con ceros
        markers = np.zeros((img.shape[0], img.shape[1]),dtype=np.int32)

        # Dibujamos los contornos encontrados en los marcadores
        for i in range(len(contours)):
            cv2.drawContours(markers, contours, i, (i+1), -1)

        # Agregamos un valor constante a los marcadores para separar los objetos
        markers = markers + 1

        # Aplicamos el método Watershed para segmentar la imagen
        markers = cv2.watershed(img, markers)

        # Coloreamos los marcadores para visualizar las regiones segmentadas
        img[markers == -1] = [255,0,0]

        # Mostramos la imagen original y la segmentada
        cv2.imshow("Original",img)
        cv2.waitKey(0)
        # Convertimos la matriz de marcadores a una imagen de escala de grises
        markers_img = cv2.convertScaleAbs(markers)

        # Aplicamos una escala de colores a la imagen de marcadores
        colormap = cv2.applyColorMap(markers_img, cv2.COLORMAP_JET)

        # Mostramos la imagen segmentada
        cv2.imshow("Segmentada", colormap)

        # Esperamos a que el usuario presione una tecla para cerrar las ventanas
        cv2.waitKey(0)
    cv2.destroyAllWindows()

metodo_watersheed()