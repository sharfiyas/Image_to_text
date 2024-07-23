import cv2
import pytesseract
img=cv2.imread('/home/user/Videos/bd_july/ocr/text/pic_handwritten_text.jpeg')
text=pytesseract.image_to_string(img)
print(text)
cv2.imshow('IMAGE',img)
cv2.waitKey(5000)
cv2.destroyAllWindows()