import cv2
cascade_file = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
def img_detect(img_gray, frame):
    faces = cascade_file.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    faces_number = len(faces)
    cv2.putText(frame, f"Gorseldeki algilanan yuz sayisi: {faces_number} ", (10, 40), cv2.FONT_HERSHEY_TRIPLEX, 1,
                (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"(programdan cikmak icin 'e' tusuna basiniz!)", (10, 90), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                (255, 0, 0), 1, cv2.LINE_AA)
    return frame

img_cap = cv2.VideoCapture(0)

while True:
    result, frame = img_cap.read()
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dtc = img_detect(img_gray, frame)
    cv2.imshow('gercek-zamanli-yuz-algilama', dtc)
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cv2.waitKey(0)
cv2.destroyAllWindows()