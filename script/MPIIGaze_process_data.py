# coding=utf-8
import cv2
import numpy as np
import os
import glob
from img_tool import *

# args
MPIIGAZE_path='D:\\winpython\\py36\\work\\eye-gaze\\MPIIGaze'
# TO DO
todo=1
# 0 test a image
TXT= "p01"
INDEX=11
# 1 output darknet dataset format
RE_size=0
OUTPUT_folder = "C:\\dataset\\gaze_o"
# 2 test output darknet txt
TEST_folder= "C:\\dataset\\gaze\\MPIIGaze_p00\\images"
TEST_name="day02_0017_right"


#Annotation Subset 左眼左 ,左眼右 ,右眼左,右眼右,嘴角左,嘴角右,左眼中 ,左眼右
original_path=os.path.join(MPIIGAZE_path,'Data','Original')
annotation_subset_path=os.path.join(MPIIGAZE_path,'Annotation Subset')



def get_landmarks(points,label):
    landmarks=np.zeros(shape=(3,2),dtype=np.int16)
    if label=='left':
        landmarks=np.vstack([ points[0],points[6],points[1]])
    if label=='right':
        landmarks=np.vstack( [points[2],points[7],points[3]])

    return landmarks

def get_crop_rect(landmarks):
    
    longSide=landmarks[2,0]-landmarks[0,0]
    border= int(round(longSide*0.08))
    #border=5 
    size=longSide+border*2
    center_x=int(round( (landmarks[2,0]+landmarks[0,0])/2.))
    center_y=int(round( (landmarks[2,1]+landmarks[0,1])/2.))
    return center_x,center_y,size

    
    
def get_rect(img,landmarks):
    rate=0.2
    length=int((landmarks[2,0]-landmarks[0,0])*rate)
    xmin=landmarks[1,0]-length
    ymin=landmarks[1,1]-length
    xmax=landmarks[1,0]+length
    ymax=landmarks[1,1]+length
    return xmin,ymin,xmax,ymax
def read_annotation(txt_path):
    with open(txt_path, 'r', encoding='utf8') as f:
        data=f.read()
        daylist=data.split('day')[1:]
        cleanlist=[]
        for aDay in daylist:
            dict={}
            dayData=aDay.split('\n')[0].split(' ')
            dict['name']='day'+dayData[0]
            dict['data']=np.array(list(map(int,dayData[1:]))).reshape(-1, 2)
            cleanlist.append(dict)
    
    return cleanlist
def get_nobj_crop_rect(center_x,center_y,landmarks):
    longSide=landmarks[2,0]-landmarks[0,0]
    border= - int(round(longSide*0.08))
    size=longSide+border*2
    pt1=(center_x-size*0.9,center_y,size)
    pt2=(center_x+size*0.9,center_y,size)
    pt3=(center_x,center_y+size*0.7,size)
    pt4=(center_x,center_y-size*0.8,size)
    return pt1,pt2,pt3,pt4

def process_dataset(txt_list):
    annotations=''
   
    for txt_file in txt_list:
        cleanlist=[]
        p_list=txt_file.split('.txt')[0].split('\\')[-1]
        cleanlist=read_annotation(txt_file)      
        for item in cleanlist:
            points=item['data']
            _file=item['name'].split('/')
            filename=os.path.split(_file[1])[1].split(".")[0]
            day_list=_file[0]
            image=cv2.imread(os.path.join(original_path,p_list,day_list,_file[1]))
            oh,ow,_=image.shape
            os.makedirs(os.path.join(OUTPUT_folder,'MPIIGaze_'+p_list,'images'),mode=0o777,exist_ok=True)
            for act in ['left','right']:
                landmarks=get_landmarks(points,act)
                center_x,center_y,crop_size=get_crop_rect(landmarks)
                act_image=crop(image,center_x,center_y,crop_size,crop_size)
                
                pt1,pt2,pt3,pt4=get_nobj_crop_rect(center_x,center_y,landmarks)
                no_image1=crop2(image,pt1)
                no_image2=crop2(image,pt2)
                no_image3=crop2(image,pt3)
                no_image4=crop2(image,pt4)

                if crop_size==RE_size or RE_size==0:
                    image_size=crop_size
                else:
                    act_image=resize(act_image,RE_size,RE_size)
                    image_size=RE_size
                    no_image1=resize(no_image1,RE_size,RE_size)
                    no_image2=resize(no_image2,RE_size,RE_size)
                    no_image3=resize(no_image3,RE_size,RE_size)
                    no_image4=resize(no_image4,RE_size,RE_size)
                offset_w=center_x-int(crop_size/2)
                offset_h=center_y-int(crop_size/2)
                
                xmin,ymin,xmax,ymax=get_rect(act_image,landmarks)
                #draw_rect(act_image,xmin-offset_w,ymin-offset_h,xmax-offset_w,ymax-offset_h)
                hsvI = cv2.cvtColor(act_image, cv2.COLOR_BGR2HSV)
                V=np.sum(hsvI[:,:,2])/crop_size/crop_size
                #if V<35:
                if V<15:
                    #print('skip %s %s'%(os.path.join(original_path,p_list,day_list,_file[1]),act))
                    os.makedirs(os.path.join(OUTPUT_folder,'skip_'+p_list),mode=0o777,exist_ok=True)
                    cv2.imwrite( os.path.join(OUTPUT_folder,'skip_'+p_list,day_list+'_'+filename+'_'+act+'.jpg'), act_image );
                    continue
                cv2.imwrite( os.path.join(OUTPUT_folder,'MPIIGaze_'+p_list,'images',day_list+'_'+filename+'_'+act+'.jpg'), act_image );
                act_image=None
                os.makedirs(os.path.join(OUTPUT_folder,'no_'+p_list,'images'),mode=0o777,exist_ok=True)
                cv2.imwrite( os.path.join(OUTPUT_folder,'no_'+p_list,'images',day_list+'_'+filename+'_'+act+'_1.jpg'), no_image1 );
                with open(os.path.join(OUTPUT_folder,'no_'+p_list,'images',day_list+'_'+filename+'_'+act+'_1.txt'), 'w', encoding='utf8') as f:
                    f.write('')
                no_image1=None
                cv2.imwrite( os.path.join(OUTPUT_folder,'no_'+p_list,'images',day_list+'_'+filename+'_'+act+'_2.jpg'), no_image2 );
                with open(os.path.join(OUTPUT_folder,'no_'+p_list,'images',day_list+'_'+filename+'_'+act+'_2.txt'), 'w', encoding='utf8') as f:
                    f.write('')
                no_image2=None
                cv2.imwrite( os.path.join(OUTPUT_folder,'no_'+p_list,'images',day_list+'_'+filename+'_'+act+'_3.jpg'), no_image3 );
                with open(os.path.join(OUTPUT_folder,'no_'+p_list,'images',day_list+'_'+filename+'_'+act+'_3.txt'), 'w', encoding='utf8') as f:
                    f.write('')
                no_image3=None
                cv2.imwrite( os.path.join(OUTPUT_folder,'no_'+p_list,'images',day_list+'_'+filename+'_'+act+'_4.jpg'), no_image4 );
                with open(os.path.join(OUTPUT_folder,'no_'+p_list,'images',day_list+'_'+filename+'_'+act+'_4.txt'), 'w', encoding='utf8') as f:
                    f.write('')
                no_image4=None



                x,y,w,h=convert(crop_size,crop_size,xmin-offset_w,ymin-offset_h,xmax-offset_w,ymax-offset_h)
                
                out_box_txt='%s %s %s %s %s'%(
                    0,
                    x,
                    y,
                    w,
                    h,
                    )
                with open(os.path.join(OUTPUT_folder,'MPIIGaze_'+p_list,'images',day_list+'_'+filename+'_'+act+'.txt'), 'w', encoding='utf8') as f:
                    f.write(out_box_txt)

                
            image=None 
            
        
                   
    """with open(os.path.join(OUTPUT_folder,'MPIIGaze_annotations.txt'), 'w', encoding='utf8') as e:
        e.write(annotations)"""
    


if __name__ == '__main__':

    if todo==0:
        cleanlist=read_annotation(os.path.join(annotation_subset_path,TXT+'.txt'))
        points=cleanlist[INDEX]['data']
        _file=cleanlist[INDEX]['name'].split('/')
        image=cv2.imread(os.path.join(original_path,TXT,_file[0],_file[1]))
        for act in ['left','right']:
            item=get_landmarks(points,act)
            draw_points(image,item)
            xmin,ymin,xmax,ymax=get_rect(image,item)
            draw_rect(image,xmin,ymin,xmax,ymax)

        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if todo==1:
        txt_list=glob.glob(annotation_subset_path + '/*.txt')
        process_dataset(txt_list)
        print('Successfully')
    if todo==2:
        image=test_a_output(TEST_folder, TEST_name)
        cv2.imshow("image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

