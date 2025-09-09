import cv2 as cv

# Load Haar Cascade classifiers
haar_path = cv.data.haarcascades
face_cascade = cv.CascadeClassifier(haar_path + 'haarcascade_frontalface_default.xml')
smile_cascade = cv.CascadeClassifier(haar_path + 'haarcascade_smile.xml')

# Open webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Show face count on screen (top-left corner)
    cv.putText(frame, f"Faces: {len(faces)}", (20, 40),
               cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # ROI for smile detection
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

        if len(smiles) > 0:
            cv.putText(frame, "Smiling :)", (x, y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Show live video
    cv.imshow('Face, Smile & Count Detection', frame)

    # Exit on 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
