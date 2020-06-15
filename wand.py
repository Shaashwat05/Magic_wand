import cv2
import numpy as np
import time
from mnist import predict

x, y, k = 200, 200, -1

kernel1=np.asarray([[1,0,-1],[1,0,-1],[1,0,-1]])
kernel2=np.asarray([[1,1,1],[0,0,0],[-1,-1,-1]])


cap = cv2.VideoCapture(0)

stp = 0

old_pts = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)

ret, frame = cap.read()
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#edges=np.sqrt(convolution(np.float32(gray_frame/255),kernel1)**2+convolution(np.float32(gray_frame/255),kernel2)**2)
#ret,thresh2 = cv2.threshold(edges*255,80,255,cv2.THRESH_BINARY)

mask = np.ones_like(frame)

time.sleep(3)
print(mask)

lower_red = np.array([30,150,50])
upper_red = np.array([255,255,180])


def command(num):
    if(num == 1):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_red, upper_red)
 

count = 0

while True:
    check, new_frame = cap.read()
    new_frame = cv2.flip(new_frame, 1)
    new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    #edges=np.sqrt(convolution(np.float32(new_gray/255),kernel1)**2+convolution(np.float32(new_gray/255),kernel2)**2)
    #ret,thresh1 = cv2.threshold(edges*255,80,255,cv2.THRESH_BINARY)
    cv2.imshow("gray", new_gray)
    new_pts, status, err = cv2.calcOpticalFlowPyrLK(gray_frame,
                                                    new_gray,
                                                    old_pts,
                                                    None, maxLevel=1,
                                                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                                              15, 0.08))

    for i, j in zip(old_pts, new_pts):
        x, y = j.ravel()
        a, b = i.ravel()
        if stp == 0:
            mask = cv2.line(mask, (a, b), (x, y), (255, 255, 255), 6)

        cv2.circle(new_frame, (x, y), 6, (0, 255, 0), -1)
    
    num, prob = predict(mask)
    if(prob > 0.7):
        filter = num

    new_frame = cv2.addWeighted(mask, 0.3, new_frame, 0.7, 0)
    new_frame = cv2.rectangle(new_frame, (0,0), (50, 50), (127, 255, 127), -1) 
    if(filter == 1):
        hsv = cv2.cvtColor(new_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_red, upper_red)
        cv2.imshow("OutPut Window", mask)
    elif(filter == 2):

    

    cv2.imshow("Result Window", mask)

    cv2.imwrite("frame1.jpg", mask)

    gray_frame = new_gray.copy()

    old_pts = new_pts.reshape(-1, 1, 2)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
