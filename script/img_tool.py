import numpy as np
import cv2
import os

def convert(ow,oh, xmin,ymin,xmax,ymax):
    dw = 1./ow
    dh = 1./oh
    x = (xmax + xmin)/2.0 - 1
    y = (ymax + ymin)/2.0 - 1
    w = xmax - xmin
    h = ymax - ymin
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
def reverse_convert(ow,oh, x,y,w,h):
    x = float(x)*ow
    w = float(w)*ow
    y = float(y)*oh
    h = float(h)*oh
    xmax= int(round(x+1+(w/2.)))
    ymax= int(round(y+1+(h/2.)))
    xmin= int(round(x+1-(w/2.)))
    ymin= int(round(y+1-(h/2.)))
    return (xmin,ymin,xmax,ymax)
def os_path(path):
    return "\\".join(path.split("/"))
def draw_rect(img,xmin,ymin,xmax,ymax):
    img= cv2.rectangle(img, (xmin,ymin),(xmax,ymax), ( 0, 0,255),thickness=1) 
def draw_points(img,landmarks):
    for item in list(landmarks):
        cv2.circle(img,(item[0],item[1]),2,(0, 255, 0),thickness=1)
def crop(img,center_x,center_y,width,height):
    xmin=int(round(center_x-width/2))
    ymin=int(round(center_y-height/2))
    xmax=int(xmin+width)
    ymax=int(ymin+height)
    return img[ymin:ymax,(xmin):(xmax)]
def crop2(img,points):
    (center_x,center_y,size)=points
    return crop(img,center_x,center_y,size,size)
def resize(img,width,height):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

def test_a_output(test_folder,test_name):
    path_jpg = os.path.join(test_folder, test_name+'.jpg')
    path_txt = os.path.join(test_folder, test_name+'.txt')
    image=cv2.imread(path_jpg)
    oh,ow,_=image.shape
    with open(path_txt) as f:
        data = f.readline()
    data = data.split(' ')
    xmin,ymin,xmax,ymax=reverse_convert(ow,oh,data[1],data[2],data[3],data[4])
    draw_rect(image,xmin,ymin,xmax,ymax)
    return image
