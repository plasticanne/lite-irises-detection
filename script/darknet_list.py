# coding=utf-8
import numpy as np
import cv2
import os
import glob
from img_tool import reverse_convert
VAILD_ratio=0.2
RANDOM=True
INPUT_folder="C:\\dataset\\gaze"
OUTPUT_list_folder="C:\\dataset"
OUTPUT_annotations=False
UNITY_list=[1,15] #(imgs count,1000 count) 
MPIIGAZE_list=15 #p count
MPIIGAZE_count=0#25000
NO_list=15
NO_count=0#40000
def os_path(path):
    return "\\".join(path.split("/"))
def process_list(jpg_list,output_file):
    out_list=''
    annotations=''
    for jpg_file in jpg_list:
        image_path= os_path(os.path.join( os.getcwd(),jpg_file))
        out_list='%s%s\n'%(out_list,image_path)
        if OUTPUT_annotations:
            image=cv2.imread(image_path)
            oh,ow,_=image.shape
            with open(image_path.split(".jpg")[0]+'.txt' ,'r', encoding='utf8') as f:
                data = f.readline()
            data = data.split(' ')
            xmin,ymin,xmax,ymax=reverse_convert(ow,oh,data[1],data[2],data[3],data[4])
            annotations=annotations+'%s %s,%s,%s,%s,%s\n'%(
                os_path(os.path.join( os.getcwd(),jpg_file)),xmin,ymin,xmax,ymax,0)
    with open(output_file, 'w', encoding='utf8') as f:
        f.write(out_list)
    if OUTPUT_annotations:
        with open(os.path.join(OUTPUT_list_folder,'annotations.txt'), 'w', encoding='utf8') as e:
            e.write(annotations)

def DoRandom(list):
    np.random.seed(106033)
    np.random.shuffle(list)
    np.random.seed(None)

if __name__ == '__main__':
    jpg_list=[]
    unity_list=[]
    for i in range(UNITY_list[0]):
        for j in range(UNITY_list[1]):
            unity_list.append('Unity_imgs'+str(i)+'_'+ str(j))
    for unity_folder in unity_list:
        jpg_list=jpg_list+glob.glob(os.path.join( INPUT_folder,unity_folder,'images') + '/*.jpg')


    mpii_list=[]
    mpii_jpg_list=[]
    for k in range(MPIIGAZE_list):
        mpii_list.append('MPIIGaze_p'+"{:0>2d}".format(k))
    for mpii_folder in mpii_list:
        mpii_jpg_list=mpii_jpg_list+glob.glob(os.path.join( INPUT_folder,mpii_folder,'images') + '/*.jpg')
    DoRandom(mpii_jpg_list)
    

    no_list=[]
    no_jpg_list=[]
    for l in range(NO_list):
        no_list.append('no_p'+"{:0>2d}".format(l))
    for no_folder in no_list:
        no_jpg_list=no_jpg_list+glob.glob(os.path.join( INPUT_folder,no_folder,'images') + '/*.jpg')
    DoRandom(no_jpg_list)

    jpg_list=jpg_list+no_jpg_list[:NO_count]+mpii_jpg_list[:MPIIGAZE_count]
    if RANDOM:
        DoRandom(jpg_list)
    len_all=len(jpg_list)
    len_vaild=int(len_all*VAILD_ratio)
    len_train=len_all-len_vaild
    train_list=jpg_list[:len_train]
    vaild_list=jpg_list[len_train:]
    #print(train_list)
    process_list(train_list,os.path.join(OUTPUT_list_folder,'yolo_train.txt'))
    if len_vaild>0:
        process_list(vaild_list,os.path.join(OUTPUT_list_folder,'yolo_vaild.txt'))