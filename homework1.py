import cv2
import numpy as np

drawing = False
ix, iy = -1, -1
mx, my = -1, -1
show_text = ""
ALPHA = 0.5

def hue_to_bgr(h):
    hsv = np.uint8([[[h, 255, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)

def nothing(x): pass

def on_mouse(event, x, y, flags, param):
    global ix, iy, mx, my, drawing, img, show_text

    mx, my = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    
        hue = cv2.getTrackbarPos('value', 'image')
        color = hue_to_bgr(hue)
        overlay = img.copy()
        cv2.rectangle(overlay, (ix, iy), (x, y), color, -1)
        cv2.addWeighted(overlay, ALPHA, img, 1-ALPHA, 0, img)

        show_text = f"Mouse Position ({ix}, {iy}) - ({x}, {y})"

img = cv2.imread('image.png', 1)
if img is None:
    img = np.zeros((512, 512, 3), np.uint8)

cv2.namedWindow('image')
cv2.setMouseCallback('image', on_mouse)
cv2.createTrackbar('value', 'image', 0, 179, nothing)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    display = img.copy()

    if drawing:
        hue = cv2.getTrackbarPos('value', 'image')
        color = hue_to_bgr(hue)
        overlay = display.copy()
        cv2.rectangle(overlay, (ix, iy), (mx, my), color, -1)
        cv2.addWeighted(overlay, ALPHA, display, 1-ALPHA, 0, display)

    if show_text:
        cv2.putText(display, show_text, (10, 50), font, 1, (255,255,255), 2, cv2.LINE_AA)

    cv2.imshow('image', display)

    if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
        break

cv2.destroyAllWindows()
