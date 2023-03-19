import cv2
import keras.models as km

model = km.load_model('./model.h5')
emotionList = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def detectEmotion(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    result = model.predict(img.reshape(-1, 48, 48, 1))
    mostConfident = -1
    emotionId = -1
    for idx, confidence in enumerate(result[0]):
        if confidence > mostConfident:
            mostConfident = confidence
            emotionId = idx

    return emotionList[emotionId], mostConfident * 100


if __name__ == "__main__":
    # start webcam
    video = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    while True:
        verdict, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(
            gray,
            1.1,
            5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # draw rectangle
        try:
            (x, y, w, h) = face[0]
            cv2.rectangle(frame, (x, y-50), (x + w, y + h), (0, 0, 255), 3)
            croppedImage = frame[y:y + h, x:x + w]
        except Exception:
            continue

        # detect emotion
        emotion, confidenceLevel = detectEmotion(croppedImage)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'{emotion} | {int(confidenceLevel)}% sure', (x, y - 10), font, 1, (255, 0, 0), 2)

        # display faces
        cv2.imshow('frame', frame)

        # extract image
        if cv2.waitKey(1) == ord('q'):
            print('quit responds')
            break

    # extract face from frame

    # feed face to model and get output
    # display output on screen
    video.release()
    cv2.destroyAllWindows()
