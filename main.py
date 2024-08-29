import cv2
import dlib as db


detector = db.get_frontal_face_detector()
predictor = db.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_face_orientation(landmarks):
    left_eye_x = (landmarks.part(36).x + landmarks.part(39).x) // 2
    right_eye_x = (landmarks.part(42).x + landmarks.part(45).x) // 2

    nose_x = landmarks.part(30).x

    if nose_x < left_eye_x and nose_x < right_eye_x:
        return "Virou para a direita"
    elif nose_x > left_eye_x and nose_x > right_eye_x:
        return "Virou para a esquerda"
    else:
        return "Rosto centralizado"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        orientation = get_face_orientation(landmarks)
        print(orientation)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
