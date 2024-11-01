import cv2

face_cascade_path = 'haarcascade_frontalface_alt.xml'
side_cascade_path = 'haarcascade_profileface.xml'

face_cascade = cv2.CascadeClassifier(face_cascade_path)
side_cascade = cv2.CascadeClassifier(side_cascade_path)

# 웹캠을 열기 (0은 기본적으로 첫 번째 연결된 카메라를 의미함)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    sides = side_cascade.detectMultiScale(gray, 1.1, 5)
    num_of_people = []

    for (x, y, w, h) in faces:
        cv2. rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 2)
        num_of_people.append({x, y, w, h})

    for (x_2, y_2, w_2, h_2) in sides:
        # 정면 얼굴과 겹치지 않는 경우에만 측면 얼굴을 표시
        if not any((x < x_2 < x + w and y < y_2 < y + h) for (x, y, w, h) in faces):
            cv2.rectangle(gray, (x_2, y_2), (x_2 + w_2, y_2 + h_2), (0, 255, 0), 2)
            num_of_people.append({x_2, y_2, w_2, h_2})

    number = len(num_of_people)
    print(number)


    # 프레임을 윈도우에 표시
    cv2.imshow('Webcam', gray)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
