import cv2
import face_recognition

haar_cascade = "haarcascade_frontalface_default.xml"

detector = cv2.CascadeClassifier(haar_cascade)

cam = cv2.VideoCapture(0)

cv2.namedWindow("video", cv2.WINDOW_NORMAL)
cv2.resizeWindow("video", 500, 300)

em_list = []
count = 0

while True:

    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    frame = cv2.resize(frame, (500, 500))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    k = cv2.waitKey(1)

    rectangles = detector.detectMultiScale(gray, scaleFactor=1.1, 
                        minNeighbors=5, minSize=(70, 70),
                        flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(rectangles) > 0:
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rectangles]
        encodings = face_recognition.face_encodings(frame, boxes)
        if len(em_list) == 0:
            em_list.append(encodings[0])
        if len(em_list) > 0:
            match = face_recognition.compare_faces(em_list, encodings[0])
            if True in match:
                count += 1
                for (top, right, bottom, left) in boxes:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)



    cv2.imshow("video", frame)

    
    



