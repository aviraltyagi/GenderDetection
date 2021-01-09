import cv2 as cv
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

model_gender = load_model(r"Model\GenderDetectionV16.h5")

classifier = cv.CascadeClassifier(r"Model\face_detector.xml")


def face_extractor(img):
    faces = classifier.detectMultiScale(img, 1.3, 5)
    rect_coord = []
    # Crop all faces found
    # The (x,y) coordinates are from the top-left corner of the image
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cropped_face = img[y:y + h, x:x + w]
        rect_coord.append([y, h, x, w, cropped_face])
    return rect_coord


def gender(cropped_face):
    cropped_face = cv.resize(cropped_face, (114, 92))
    im = Image.fromarray(cropped_face, 'RGB')
    img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)
    pred = model_gender.predict(img_array)[0][1]
    return pred


def open_webcam():
    cap = cv.VideoCapture(0)

    while True:
        _, frame = cap.read()

        if not _:
            break
        else:
            face = face_extractor(frame)

            for y, h, x, w, cropped_face in face:
                if type(cropped_face) is np.ndarray:
                    sex_pred = gender(cropped_face)
                    # sex_text = ''
                    if sex_pred > 0.5:
                        sex_text= 'Male'
                        sex_pred= sex_pred
                    else:
                        sex_text= 'Female'
                        sex_pred= 1-sex_pred
                    # sex_text = 'Male' if sex_pred > 0.5 else 'Female'
                    cv.putText(frame, 'Sex: {}({:.2f})'.format(sex_text, sex_pred), (x, y - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
            ret, buffer = cv.imencode(".jpg", frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def end_recording():
    cv.waitKey()
    cv.destroyWindow()
