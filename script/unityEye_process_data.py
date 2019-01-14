# coding=utf-8
import numpy as np
import cv2
import os
import ujson
import glob
from img_tool import *

# args
IMAGES_folder = "D:\\winpython\\py36\\work\\UnityEyes_Windows\\imgs0"
# TO DO
todo=1
# 0 test a image
NAME="1"
# 1 output darknet dataset format
CROP_size=416
RE_size=416
OUTPUT_folder = "C:\\dataset\\gaze_o\\Unity_imgs0"
# 2 test output darknet txt
TEST_folder= "C:\\dataset\\gaze_o\\Unity_imgs0_0\\images"
TEST_name="1_flip"





def process_dataset(json_list,output_folder):
    annotations=''
    for json_file in json_list:
        filename=os.path.split(json_file)[1].split(".")[0]
        image_path= os.path.join(IMAGES_folder,filename+'.jpg')
        image=cv2.imread(image_path)
        oh,ow,_=image.shape
        image=crop(image,ow/2,oh/2,CROP_size,CROP_size)
        if CROP_size==RE_size:
            image_size=CROP_size
        else:
            image=resize(image,RE_size,RE_size)
            image_size=RE_size
        image_flip=cv2.flip(image, flipCode=1)
        cv2.imwrite( os.path.join(output_folder,filename+'.jpg'), image );
        cv2.imwrite( os.path.join(output_folder,filename+'_flip.jpg'), image_flip );
        iris_2d=get_landmarks(json_file,'iris_2d',ow,oh,offset_w=-int((ow-CROP_size)/2.),offset_h=-int((oh-CROP_size)/2.))
        xmin,ymin,xmax,ymax=get_rect(image,iris_2d)
        x,y,w,h=convert(CROP_size,CROP_size, xmin,ymin,xmax,ymax)
        out_box_txt='%s %s %s %s %s'%(
                    0,
                    x,
                    y,
                    w,
                    h,
            )
        with open(os.path.join(output_folder,filename+'.txt'), 'w', encoding='utf8') as f:
            f.write(out_box_txt)


        flip_out_box_txt='%s %s %s %s %s'%(
                    0,
                    1-x,
                    y,
                    w,
                    h,
            )
        with open(os.path.join(output_folder,filename+'_flip.txt'), 'w', encoding='utf8') as g:
            g.write(flip_out_box_txt)

        
        """annotations=annotations+'%s %s,%s,%s,%s,%s\n'%(
                    os_path(os.path.join(output_folder,filename+'.jpg')),xmin,ymin,xmax,ymax,0)
        annotations=annotations+'%s %s,%s,%s,%s,%s\n'%(
                    os_path(os.path.join(output_folder,filename+'_flip.jpg')),image_size-xmax,ymin,image_size-xmin,ymax,0)           
    with open(os.path.join(OUTPUT_folder,'Unity_annotations.txt'), 'w', encoding='utf8') as e:
        e.write(annotations)"""
def get_landmarks(path,target,ow,oh,iw=0,ih=0,offset_w=0,offset_h=0):
    with open(path, 'r', encoding='utf8') as f:
        json_data = ujson.load(f)
    rateh=ih/oh if ih>0 else 1
    ratew=iw/ow if iw>0 else 1
    landmarks=np.zeros([len(json_data[target]),2], dtype=np.int16)
    for i, val in enumerate(json_data[target]):
        #landmarks[i,:2]=np.asarray(eval(np.asarray(json_data[target])[i]))[:2]
        new=np.array([np.int_(eval(val)[0]*ratew+offset_w),np.int_((oh-eval(val)[1])*rateh+offset_h)])
        landmarks[i,:2]=new
    return landmarks
def get_rect(img,landmarks):
    # get rect from eye landmarks
    border = 5
    l_ul_x = min(landmarks[:,0])
    l_ul_y = min(landmarks[:,1])
    l_lr_x = max(landmarks[:,0])
    l_lr_y = max(landmarks[:,1])
    #print(l_ul_x, l_ul_y,l_lr_x,l_lr_y)
    long= max(l_lr_y-l_ul_y,l_lr_x-l_ul_x)
    paddingX=np.int_(np.sum(long-(l_lr_x-l_ul_x))/2)
    paddingY=np.int_(np.sum(long-(l_lr_y-l_ul_y))/2) 
    paddingX=0
    paddingY=0
    
    xmin = int(round(np.sum(l_ul_x)))-border-paddingX
    ymin = int(round(np.sum(l_ul_y)))-border-paddingY
    xmax = int(round(np.sum(l_lr_x)))+border+paddingX
    ymax = int(round(np.sum(l_lr_y)))+border+paddingY
    return xmin,ymin,xmax,ymax

if __name__ == '__main__':

    if todo==0:
        path_jpg = os.path.join(IMAGES_folder, NAME+'.jpg')
        path_json = os.path.join(IMAGES_folder, NAME+'.json')
        image=cv2.imread(path_jpg)
        oh,ow,_=image.shape
        caruncle_2d=get_landmarks(path_json,'caruncle_2d',ow,oh)
        draw_points(image,caruncle_2d)
        interior_margin_2d=get_landmarks(path_json,'interior_margin_2d',ow,oh)
        draw_points(image,interior_margin_2d)
        iris_2d=get_landmarks(path_json,'iris_2d',ow,oh)
        draw_points(image,iris_2d)
        xmin,ymin,xmax,ymax=get_rect(image,iris_2d)
        draw_rect(image,xmin,ymin,xmax,ymax)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if todo==1:
        json_list=glob.glob(IMAGES_folder + '/*.json')
        for i in range(0,int(len(json_list)/1000)):
            os.makedirs(os.path.join(OUTPUT_folder+'_'+str(i),'images'),mode=0o777,exist_ok=True)
            if (i+1)*1000<len(json_list):      
                process_dataset(json_list[i*1000:(i+1)*1000],os.path.join(OUTPUT_folder+'_'+str(i),'images'))
            if (i+1)*1000>len(json_list):
                process_dataset(json_list[i*1000:],os.path.join(OUTPUT_folder+'_'+str(i),'images'))
        print('Successfully')
    if todo==2:
        image=test_a_output(TEST_folder, TEST_name)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
 