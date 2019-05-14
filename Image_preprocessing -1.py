"""This program is to be used to detect and extract only the faces of the students from the given images."""
"""For the datastructure refer to the example."""

"""This code uses the new dataset of celebrities imported"""

"""The code is supposed to use face detection to detect the face and replace the original image by just the face"""

# Importing the libraries
import os
import cv2
import sys
import dlib


def extract_faces(filepath,endpath,start_folder=0):
    """This function is used to detect and extract the faces from the input imges"""

    """Filepath is the initial location of the images. For the datastructure of the stored images look at the example."""

    sub_folders = os.listdir(filepath)
    for j in range(start_folder,len(sub_folders)) :
        os.makedirs(endpath+"/training-data/"+sub_folders[j],exist_ok=True)
        os.makedirs(endpath+"/test-data/" + sub_folders[j], exist_ok=True)
        for i,image in enumerate(os.listdir(os.path.join(filepath,sub_folders[j]))):
            image_path = filepath + "/" + sub_folders[j] + "/" + image
            img = cv2.imread(image_path)
            try:
                # x1= face_detection(img).left()
                # x2 = face_detection(img).right()
                # y1 = face_detection(img).top()
                # y2 = face_detection(img).bottom()
                coods = face_detection(img)
                # x,y,w,h=face_detection(img)
                if i < 10: # Change this number to change the number of images in training set and test set
                    cv2.imwrite(endpath+"/training-data/" + sub_folders[j] + "/" + sub_folders[j] + "_" + str(
                            i + 1) + ".jpg", img[coods[1]:coods[3],coods[0]:coods[2],])
                    # im = cv2.imread(endpath+"training-data/" + sub_folders[j] + "/" + sub_folders[j] + "_" + str(i + 1) + ".jpg")
                    # cv2.imshow("Cropped",im)
                    # cv2.waitKey()
                    # cv2.destroyAllWindows()
                else:
                    cv2.imwrite(
                            endpath+"/test-data/" + sub_folders[j] + "/" + sub_folders[j] + "_" + str(i + 1) + ".jpg",
                        img[coods[1]:coods[3], coods[0]:coods[2],])
            except IndexError:

                sys.exit("Please provide another image.\nImage error :" + filepath+"/"+sub_folders[j]+"/"+image)




def face_detection(image):
    """Function to detect prominent faces from the given images"""

    # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # # If you fail to get the desired face or any face just tweak parameters minNeighbours and the training data (.xml) file
    # harr_classifier1 = cv2.CascadeClassifier("C:\Anaconda3\envs\gpu_env\Library\etc\haarcascades\haarcascade_frontalface_alt.xml")
    # # try:
    # #     faces = harr_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
    # #     if faces<0:
    # #         raise ValueError("Please input a clearer image with proper illumination")
    # #     else:
    # # # x, y, w, h = faces[0]
    # #         return faces[0]
    # faces = harr_classifier1.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
    # return faces[0]
    # # while True:
    # #     try:
    # #         faces = harr_classifier1.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)
    # #         return faces[0]
    # #     except IndexError:
    # #         cv2.imshow("Image",resize(image))
    # #         cv2.waitKey(2000)
    # #         sys.exit("Please input a clearer image with proper illumination.")
    hogFaceDetector = dlib.get_frontal_face_detector()
    faceRects = hogFaceDetector(image, 0)
    coods = []
    for faceRect in faceRects:
        x1 = faceRect.left()
        y1 = faceRect.top()
        x2 = faceRect.right()
        y2 = faceRect.bottom()
        coods = [x1,y1,x2,y2]
    return coods

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

extract_faces('dataset_new_celebs','celebs_extracted')