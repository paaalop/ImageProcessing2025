import cv2
import numpy as np

MIN_MATCH_COUNT = 10
IMAGE_PATH = 'dongbaek.jpg'
VIDEO_PATH = 'dongbaek.mp4'

MAX_WIDTH = 640
MAX_HEIGHT = 480

img1_orig = cv2.imread(IMAGE_PATH, 0) # queryImage (흑백)

h_orig, w_orig = img1_orig.shape[:2]
ratio_q = min(MAX_WIDTH / w_orig, MAX_HEIGHT / h_orig) 
if ratio_q < 1:
    new_w_q, new_h_q = int(w_orig * ratio_q), int(h_orig * ratio_q)
    img1 = cv2.resize(img1_orig, (new_w_q, new_h_q), interpolation=cv2.INTER_AREA)
else:
    img1 = img1_orig

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
h, w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)

cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame_color_orig = cap.read() 
    
    if not ret:
        break
    h_f, w_f = frame_color_orig.shape[:2]
    ratio_f = min(MAX_WIDTH / w_f, MAX_HEIGHT / h_f)
    
    if ratio_f < 1:
        new_w_f, new_h_f = int(w_f * ratio_f), int(h_f * ratio_f)
        frame_resized_color = cv2.resize(frame_color_orig, (new_w_f, new_h_f), interpolation=cv2.INTER_AREA)
    else:
        frame_resized_color = frame_color_orig

    img2 = cv2.cvtColor(frame_resized_color, cv2.COLOR_BGR2GRAY) 

    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is not None and des2 is not None and len(des2) > 1 and len(des1) > 1:

        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            if M is not None:
                dst = cv2.perspectiveTransform(pts, M)
                
                frame_final = cv2.polylines(frame_resized_color.copy(), [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                frame_final = frame_resized_color.copy()
        else:
            matchesMask = None
            frame_final = frame_resized_color.copy()
            
        draw_params = dict(matchColor = (0,255,0), singlePointColor = None, 
                            matchesMask = matchesMask, flags = 2)
        
        img3 = cv2.drawMatches(img1, kp1, frame_final, kp2, good, None, **draw_params)
        
        cv2.imshow('Feature Matching and Tracking (Resized)', img3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
