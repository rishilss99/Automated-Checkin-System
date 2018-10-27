import cv2
import os
import numpy as np

def resize(image):
    img = cv2.resize(image,(700,700),interpolation=cv2.INTER_LANCZOS4)
    return img

subjects = ["","Rishil"]

def face_detection(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    harr_classifier = cv2.CascadeClassifier("E:\Rishil's Documents\Python Programs 3\Computer Vision(Drishti Project)\Mastering OpenCV in Python(Udemy)\Master OpenCV\Haarcascades\haarcascade_frontalface_default.xml")
    faces = harr_classifier.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
    x,y,w,h = faces[0]
    return gray[y:y+w,x:x+h],faces[0]
# cv2.imshow("Image",image)
# cv2.waitKey()
# cv2.destroyAllWindows()

def prepare_dataset(folder):
    dirs = os.listdir(folder)
    faces=[]
    labels=[]
    for dir_name in dirs:
        label = int(dir_name.replace("s", ""))
        subject_dir_path = folder + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            image_path = subject_dir_path + "/" + image_name
            image = cv2.imread(image_path)
            # image = resize(image)  # Added by me
            face, rect = face_detection(resize(image))
            face = resize(face)  # Added by me
            faces.append(resize(face))
            labels.append(label)
    return faces,labels
faces,labels=prepare_dataset("training-data")
# print(len(faces),len(labels))
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.train(faces,np.array(labels))

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def predict(test_img):
    img=test_img.copy()
    face,rect=face_detection(img)
    face = resize(face) # Added by me
    label = face_recognizer.predict(face)
    label_text = subjects[label[0]]
    draw_rectangle(img,rect)
    draw_text(img,label_text,rect[0],rect[1]-5)
    return img
test_img1 = cv2.imread("test-data/test1.jpg")
# # test_img1 = resize(test_img1) # Added by me
predicted_img1 = predict(test_img1)
cv2.imshow(subjects[1],resize(predicted_img1))
cv2.waitKey()
cv2.destroyAllWindows()

