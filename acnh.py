import cv2
import imutils
import numpy as np
import pytesseract

f_width = 1280
f_height = 720
catalog = set()

def createMask() :
    mask = np.zeros((f_height, f_width), dtype=np.uint8)
    mask[int(f_height*.215):int(f_height*.885),int(f_width*.495):int(f_width*.835)] = 1
    return mask

def addToCatalog(text) :
    for line in text.splitlines() :
        item = line.strip()
        if item is not '' :
            catalog.add(item)
            print (item)

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, f_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, f_height)
catalog_mask = createMask()

while(True):

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=catalog_mask)

    addToCatalog(pytesseract.image_to_string(masked))

    cv2.imshow('Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
