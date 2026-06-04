import cv2
import numpy as np

cam = cv2.VideoCapture(1)

if not cam.isOpened():
    print("No se pude abrir la camara")
    exit()
while True:
    ret, frame = cam.read()
    if not ret:
        print("No se puede recibir la imagen")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detector de bordes Kanny
    frame1 = cv2.Canny(frame, 100, 200)
    
    # Nose, lo hizo claudio
    kernel = np.ones((3, 3), np.uint8)
    frame1 = cv2.dilate(frame1, kernel, iterations = 1)
    
    #Mostrar imagen
    cv2.imshow('WebCam Kanny', frame1)
    if cv2. waitKey(1) == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()