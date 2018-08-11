import cv2
import numpy as np
import numpy.linalg as la
#from umucv.kalman import kalman

def kalman(mu,P,F,Q,B,u,z,H,R):
    # mu, P : current state and its uncertainty
    # F, Q  : Dynamic system and its noise
    # B, u  : control model and the entrance
    # z     : observation
    # H, R  : Observation model and its noise
    
    mup = F @ mu + B @ u
    pp  = F @ P @ F.T + Q

    zp = H @ mup

    # if there is no observation we only do prediction

    if z is None:
        return mup, pp, zp

    epsilon = z - zp

    k = pp @ H.T @ la.inv(H @ pp @ H.T +R)

    new_mu = mup + k @ epsilon
    #print(new_mu)
    new_P  = (np.eye(len(P))-k @ H) @ pp
    return new_mu, new_P, zp

cap = cv2.VideoCapture('2 (2).mp4')
loHue = 0
loSaturation = 50
loValue = 50
high_hue = 0
high_saturation = 255
high_value = 255

def low_hue(x):
 	global loHue
 	loHue = x 


def upper_hue (x):
	global high_hue
	high_hue = x



cv2.namedWindow('Trackbars', flags=cv2.WINDOW_OPENGL)
cv2.resizeWindow('Trackbars', 500, 30)
cv2.moveWindow('Trackbars', 500, 600)
cv2.createTrackbar('loHue', 'Trackbars', 0, 180, low_hue)

cv2.createTrackbar('upperHue', 'Trackbars', 0, 180, upper_hue)

cv2.setTrackbarPos('loHue', 'Trackbars', 3)
cv2.setTrackbarPos('upperHue', 'Trackbars', 9)



fps = 30
dt = 1 / fps
#t = np.arange(0, 2.01, dt)
noise = 3

a = np.array([0, 300])
F = np.array(
	[1, 0, dt, 0,
	0, 1, 0, dt,
	0, 0, 1, 0,
	0, 0, 0, 1 ]).reshape(4,4)
	

B = np.array(
	[dt**2/2, 0,
	0, dt**2/2,
	dt, 0,
	0, dt ]).reshape(4,2)
	

H = np.array(
	[1,0,0,0,
	0,1,0,0]).reshape(2,4)


mu = np.array([0,0,0,0])
P = np.diag([1000, 1000, 1000, 1000])**2


sigmaM = 0.0001
sigmaZ = 3*noise

Q = sigmaM**2 * np.eye(4)
R = sigmaZ**2 * np.eye(2)

listCenterX=[]
listCenterY=[]
xe = []
xu = []
ye = []
yu = []
xp = []
yp = []
xpu = []
ypu = []


while(True):
	_, image = cap.read()
	if _ == False:
		break
	height, width, __ = image.shape
	print(image.shape)
	roi_vertices = [
	(width / 4, height / 4.7),
	(width / 3.8, height / 1.8),
	(width / 1.64, height / 1.52),
	(width / 1.8, height / 4)

	]
	vertices = np.array([roi_vertices], np.int32)

	mask1 = np.zeros_like(image)
	mask_colour = (255,255,255)
	cv2.fillPoly(mask1, vertices, mask_colour)
	masked_image = cv2.bitwise_and(image, mask1)



	blur_masked_image = cv2.GaussianBlur(masked_image, (3, 3), 2)
	hsv = cv2.cvtColor(blur_masked_image, cv2.COLOR_BGR2HSV)
	lower_limit = np.array([loHue,loSaturation,loValue])
	upper_limit = np.array([high_hue,high_saturation,high_value])
	mask2 = cv2.inRange(hsv, lower_limit, upper_limit)
	res = cv2.bitwise_and(image, blur_masked_image, mask = mask2)
	erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	erosion = cv2.erode(mask2, erode_element, iterations = 1)

	erosion = cv2.medianBlur(erosion,3)
	dilation = cv2.dilate(erosion, dilate_element, iterations = 2)
	copy_dilation = dilation.copy()
	
	_, contours, hierarchy = cv2.findContours(copy_dilation, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	center = None

	if len(contours) > 0:
		c = max(contours, key = cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		mu,P,pred = kalman(mu,P,F,Q,B,a,np.array([x,y]),H,R)
		xe.append(mu[0])
		ye.append(mu[1])
		xu.append(2 * np.sqrt(P[0, 0]))
		yu.append(2 * np.sqrt(P[1, 1]))
		P2 = P
		mu2 = mu
		res2 = []
		for __ in range (fps * 2):
			mu2, P2, pred2 = kalman(mu2, P2, F, Q, B, a, None, H, R)
			xp.append(mu2[0])
			yp.append(mu2[1])
			xpu.append(2 * np.sqrt(P[0, 0]))
			ypu.append(2 * np.sqrt(P[1, 1]))
		for n in range(len(xp)):
			uncertainity_in_state = (xpu[n] + ypu[n]) / 2
			cv2.circle(image, (int(xp[n]), int(yp[n])),  int(uncertainity_in_state), (0,0,255))
		cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 255), -2)	
		cv2.imshow('tracking', image)


	k = cv2.waitKey(500) & 0xFF
	if k ==27:
		break
cap.release()
cv2.destroyAllWindows()