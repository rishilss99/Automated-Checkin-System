"""This program is to be used to detect and extract only the faces of the students from the given images."""
"""For the datastructure refer to the example."""

"""Minimum 6 images to be provided by the student"""

"""The code is supposed to use face detection to detect the face and replace the original image by just the face"""

# Importing the libraries
import os
import cv2
import sys

def extract_faces(filepath,start_folder=0):
    """This function is used to detect and extract the faces from the input images"""

    """Filepath is the initial location of the images. For the datastructure of the stored images look at the example."""

    sub_folders = os.listdir(filepath)
    for j in range(start_folder,len(sub_folders)) :
        os.makedirs("dataset_extracted/training-data/"+sub_folders[j],exist_ok=True)
        os.makedirs("dataset_extracted/test-data/" + sub_folders[j], exist_ok=True)
        for i,image in enumerate(os.listdir(os.path.join(filepath,sub_folders[j]))):
            image_path = filepath + "/" + sub_folders[j] + "/" + image
            img = cv2.imread(image_path)
            try:
                x,y,w,h=face_detection(img)
                if i < 4:
                    cv2.imwrite("dataset_extracted/training-data/" + sub_folders[j] + "/" + sub_folders[j] + "." + str(
                            i + 1) + ".jpg", img[y:y + w, x:x + h])
                else:
                    cv2.imwrite(
                            "dataset_extracted/test-data/" + sub_folders[j] + "/" + sub_folders[j] + "." + str(i + 1) + ".jpg",
                            img[y:y + w, x:x + h])
            except IndexError:

                sys.exit("Please input atleast 6 image with proper illumination.\nImage error :" + filepath+"/"+sub_folders[j]+"/"+image)




def face_detection(image):
    """Function to detect prominent faces from the given images"""

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # If you fail to get the desired face or any face just tweak parameters minNeighbours and the training data (.xml) file
    harr_classifier1 = cv2.CascadeClassifier("C:\Anaconda3\envs\gpu_env\Library\etc\haarcascades\haarcascade_frontalface_alt.xml")
    # try:
    #     faces = harr_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
    #     if faces<0:
    #         raise ValueError("Please input a clearer image with proper illumination")
    #     else:
    # # x, y, w, h = faces[0]
    #         return faces[0]
    faces = harr_classifier1.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
    return faces[0]
    # while True:
    #     try:
    #         faces = harr_classifier1.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
    #         return faces[0]
    #     except IndexError:
    #         cv2.imshow("Image",resize(image))
    #         cv2.waitKey(2000)
    #         sys.exit("Please input a clearer image with proper illumination.")

# def draw_rectangle(img, rect):
#     x, y, w, h= rect
#     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 7)

def resize(image):
    img = cv2.resize(image,(700,700),interpolation=cv2.INTER_LANCZOS4)
    return img


# image = cv2.imread(r"input_data_temp\vishwa\vishwa.3.jpg")
# face = face_detection(image)
# draw_rectangle(image,face)
# cv2.imshow("Image",resize(image))
# cv2.waitKey()
# cv2.destroyAllWindows()

extract_faces('input_data_temp')