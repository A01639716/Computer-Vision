import numpy as np
import cv2 
import os

images = ['road0.png','road1.png','road10.png','road100.png','road101.png','road102.png','road103.png','road104.png','road105.png','road106.png']

#Metodos de segmentacion
def metodo_watersheed(images):   
    #Parte 1 para cargar las imagenes y hacer la escala de grises 
 
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

def K_means(images): 
    for i in images:
        img = cv2.imread(i)
        img_float = np.float32(img) / 255.0

        # Definimos los parámetros del método k-means
        num_clusters = 4
        max_iteraciones = 10
        epsilon = 1.0

        # Aplicamos el método k-means
        resultado = cv2.kmeans(img_float.reshape(-1, 3), num_clusters, None, 
                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iteraciones, epsilon), 
                            attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

        # Obtenemos las etiquetas de cada pixel y los centroides de cada cluster
        labels = resultado[1]
        centroids = resultado[2]

        # Asignamos a cada pixel el valor del centroide de su cluster correspondiente
        segmentado = centroids[labels.flatten()].reshape(img.shape)

        # Convertimos la imagen segmentada a valores enteros entre 0 y 255
        segmentado = np.uint8(segmentado * 255)

        # Mostramos la imagen original y la imagen segmentada
        cv2.imshow("Original", img)
        cv2.imshow("Segmentada", segmentado)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#ruido Gausiano mas todos los metodos
def ruido_Gaus(Iruido):
    seg = input('Ingrese el metodo de segmentacion que desee 1-k-means 2- 3- 4-watersheed: ')
    for i in Iruido:
        img = cv2.imread(i)
        img1 = cv2.convertScaleAbs(img)
        # Convertimos la imagen a escala de grises
        gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        ruido = np.zeros(gray.shape, np.int16)
        #ruido Gausiano
        cv2.randn(ruido,0,25)

        #Ruido gausiano
        Gesc_5 = cv2.add(gray,np.array(ruido *0.05, dtype = np.uint8))
        Gesc_10 = cv2.add(gray,np.array(ruido *0.1, dtype = np.uint8))
        Gesc_20 = cv2.add(gray,np.array(ruido *0.2 , dtype = np.uint8))


        # Creamos una lista de nombres de archivo válidos
        filenames = [f"{i}_{j}.png" for j in range(3)]
        # Guardamos las imágenes con nombres de archivo válidos
        cv2.imwrite(filenames[0], Gesc_5)
        cv2.imwrite(filenames[1], Gesc_10)
        cv2.imwrite(filenames[2], Gesc_20)

        # Llamamos a metodo_watersheed con los nombres de archivo válidos
        if seg == '1':
            input("imagenes con diferente % de ruido")
            K_means(filenames)
        elif seg == '2':
            'x'
        elif seg == '3':
            'x'
        elif seg == '4':
            metodo_watersheed(filenames)     
    
#metodo_watersheed(images)
#K_means(images)
ruido_Gaus(images)