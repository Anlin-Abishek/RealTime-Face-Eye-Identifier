import cv2


person_name = input("Enter the name of the person to detect: ")


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Check if cascades loaded properly
if face_cascade.empty() or eye_cascade.empty():
    print("Error loading cascade files")
    exit()


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("Press ESC to exit")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    gray = cv2.equalizeHist(gray)


    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

 
    for (x, y, w, h) in faces:

      
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

       
        cv2.putText(
            frame,
            person_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA
        )

        
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

       
        eyes = eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            minSize=(15, 15)
        )

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (0, 127, 255),
                2
            )

    
    cv2.imshow("Face and Eye Detection", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
