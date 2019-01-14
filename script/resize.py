import cv2
import os
IMAGE = "./gaze/105.jpg"

LIST=[52,64,30,100,39,48,20]


if __name__ == '__main__':
    image=cv2.imread(IMAGE)
    filename=os.path.split(IMAGE)[1].split(".")[0]
    for size in LIST:
        img=cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite( filename+'_'+str(size)+'.jpg', img )