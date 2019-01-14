
#Information
This model is low cost iris position detection solution under darknet(Yolov3) framework and trained with [Unity Eyes](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/)
 and [MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/).

The [Unity Demo](https://drive.google.com/file/d/16A53pkTlKHiKDgsfmdI8-GwwyiSIYpao/view?usp=sharing) with dlib face 68 landmarks and running on opencv dnn module by an **intel i5 cpu** could be almost 60FPS (with 30FPS webcam input).

The input image is for each eye with croped near zone rect with 24*24 low resolution.

# dataset
[Unity Eyes](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/)
[MPIIGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/)
[Eye Gaze](https://www.kaggle.com/4quant/eye-gaze)

# dependents
###training on darknet
[darknet](https://pjreddie.com)

###dataset tools
    python==3.5.6
    opencv>=3.1.0
    numpy
    ujson

# Results
All the cfg and weights: [Download](https://drive.google.com/open?id=1srQwcy_Il9quS6DOIBoEvb99YLm9BEPO)

**gaze3_6_4/gaze-tiny_4_5500.weights is the best result with 24x24 small input**

gaze3_6_4/gaze-tiny_4_5500.weights
```
 detections_count = 16334, unique_truth_count = 7877
 class_id = 0, name = iris,       ap = 89.93 %
 for thresh = 0.25, precision = 0.92, recall = 0.95, F1-score = 0.93
 for thresh = 0.25, TP = 7445, FP = 611, FN = 432, average IoU = 63.49 %

 mean average precision (mAP) = 0.899301, or 89.93 %
 Total Detection Time: 19.000000 Seconds
```
gaze3_6_5/gaze-tiny_4_5500.weights
```
 detections_count = 16141, unique_truth_count = 7877
 class_id = 0, name = iris,       ap = 89.83 %
 for thresh = 0.25, precision = 0.92, recall = 0.94, F1-score = 0.93
 for thresh = 0.25, TP = 7435, FP = 613, FN = 442, average IoU = 62.29 %

 mean average precision (mAP) = 0.898293, or 89.83 %
 Total Detection Time: 20.000000 Seconds
```
gaze3_8/gaze-tiny_3_3000.weights
```
 detections_count = 15938, unique_truth_count = 7877
 class_id = 0, name = iris,       ap = 88.19 %
 for thresh = 0.25, precision = 0.89, recall = 0.91, F1-score = 0.90
 for thresh = 0.25, TP = 7148, FP = 862, FN = 729, average IoU = 58.25 %

 mean average precision (mAP) = 0.881941, or 88.19 %
 Total Detection Time: 32.000000 Seconds
```
gaze3_9/gaze-tiny_3_3000.weights
```
 detections_count = 15542, unique_truth_count = 7877
 class_id = 0, name = iris,       ap = 89.58 %
 for thresh = 0.25, precision = 0.91, recall = 0.94, F1-score = 0.92
 for thresh = 0.25, TP = 7379, FP = 704, FN = 498, average IoU = 63.05 %

 mean average precision (mAP) = 0.895787, or 89.58 %
 Total Detection Time: 33.000000 Seconds
```