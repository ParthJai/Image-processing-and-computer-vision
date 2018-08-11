# Commented part is alternative to the other part. It may or may not work well as compared to rest of the part.
import cv2
import numpy as np
#from matplotlib import pyplot as py
import math 

cap = cv2.VideoCapture('view.mp4') #put 0 as argument instead of video name to capture from webacam, live

while cap.isOpened():
    ret, coloured_frame = cap.read()

    gray_frame = cv2.cvtColor(coloured_frame, cv2.COLOR_BGR2GRAY)  #convert frame to grayscale
    #frame = frame[400:720, 550:750]  #part wherein tracks actually lie
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 2)  #renove noise
    edges = cv2.Canny(gray_frame, 120, 170)   #run canny edge detector to detect tracks
    #mask_structure = np.zeros_like(gray_frame)  #define mask size
    #mask_structure[400:720, 550:750] = 255  #render the track part white
    #mask = cv2.bitwise_and(edges, mask_structure)  #mask = frame except that mask has nothing except tracks


    height, width= gray_frame.shape   #define vertices of traingular mask
    roi_vertices = [
    (width / 2.8, height),
    (width / 2.1, height / 2.2),
    (width / 1.6, height)
    ]

    vertices = np.array([roi_vertices], np.int32)
    mask = np.zeros_like(gray_frame) #mask is same size as the frame and is filled with black colour
    mask_colour = 255  
    cv2.fillPoly(mask, vertices, mask_colour)  #fill the triangular portion with white colour while other part is black
    masked_image = cv2.bitwise_and(mask, edges)    #now mask only has tracks
    
    #(im2, contours, hier)  = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #find contours in frame, may not work as there are no "closed" curves
    #cnts = sorted(contours, key = cv2.contourArea, reverse = True)[:10]  #select 10 most strongest curves
    #screenCnt = None
#for c in cnts:
	#approximate the contour
    #peri = cv2.arcLength(c, True)
    #approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	#if our approximated contour has four points, then
	# we can assume that we have found our screen
    
    
    #if len(approx) == 2:  #select curve with 2 lines (not gonna work !)
     #   screenCnt = approx
      #  break   
    #cv2.drawContours(frame, [screenCnt], -1, (0, 255, 0), 3)  #draw contours on frame


    #sobelx_edge = cv2.Sobel(frame, -1, 1, 0, ksize = 5)  #use sobel operator instead of canny
    #sobely_edge = cv2.Sobel(frame, -1, 0, 1, ksize = 5)
    #cv2.imshow('sobelx', sobelx_edge)
    #py.figure()  #dont forget to import matplotlib!
    
    
    lines = cv2.HoughLinesP(masked_image,rho = 1,theta = np.pi/180,threshold = 85, minLineLength = 70, maxLineGap = 50)  #perform probabilistic hough transform, may require some tweaks.
    #if lines is not None:
     #   for i in range(0, len(lines)):
      #      rho = lines[i][0][0]
       #     theta = lines[i][0][1]
        #    a = math.cos(theta)  #transform polar co-ordinates to rectangulars
         #   b = math.sin(theta)
          #  x0 = a * rho
           # y0 = b * rho
            #pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))  # run 1000 pizels in both vertical and horizontal direction
            #pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            #cv2.line(coloured_frame, pt1, pt2, (255,0,0), 1, cv2.LINE_AA) #draw line according to co-ordinates above
    if lines is not None:
        for line in lines:
            for (x1, y1, x2, y2) in line:
                cv2.line(coloured_frame, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)  #draw blue coloured line of width 2 pixels from (x1,y1) to (x2,y2)



    cv2.imshow('frame', coloured_frame)

 
    if cv2.waitKey(20) & 0xff == ord('e'):  #fps = 20 and press e to exit
        break
        

        
cap.release()  #DO NOT OMIT while streaming from webcam
cv2.destroyAllWindows()
    