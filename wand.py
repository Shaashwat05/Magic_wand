import cv2
import numpy as np
import time

x, y, k = 100, 100, -1

cap = cv2.VideoCapture(0)

stp = 0

old_pts = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)

ret, frame = cap.read()
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

mask = np.ones_like(frame)

print(mask)

while True:
    check, new_frame = cap.read()
    new_frame = cv2.flip(new_frame, 1)
    new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.blur(new_gray,(7,7))
    new_gray = cv2.blur(new_gray,(7,7))
    new_gray = cv2.blur(new_gray,(7,7))
    new_gray = cv2.blur(new_gray,(7,7))
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

    new_frame = cv2.addWeighted(mask, 0.3, new_frame, 0.7, 0)
    cv2.imshow("OutPut Window", new_frame)
    cv2.imshow("Result Window", mask)

    cv2.imwrite("frame1.jpg", mask)

    gray_frame = new_gray.copy()
    old_pts = new_pts.reshape(-1, 1, 2)
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
