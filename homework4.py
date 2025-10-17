import numpy as np
import cv2

def find_and_draw_coins(image_path):
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"이미지를 읽을 수 없음")
        return

    roi = frame.copy()
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray_blur = cv2.GaussianBlur(gray, (11, 11), 0)

    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 5)
    
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    cont_img = closing.copy()
    contours, _ = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    coin_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        MIN_AREA = 50
        MAX_AREA = 10000
        
        if area < MIN_AREA or area > MAX_AREA:
            continue

        if len(cnt) < 5:
            continue

        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(roi, ellipse, (0, 255, 0), 4)
        coin_count += 1

    cv2.imshow("Original Image with Coins", roi)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_file = '.\Image0\sIMG_8253.jpg' 
    
    find_and_draw_coins(image_file)