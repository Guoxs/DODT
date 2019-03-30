### Outline of 《Toward High Performance of 3D Object Tracking in autonomous driving》

#### ① Abstract 
Multiple object tracking is a fundamental tasks in perception systems for autonomous driving, despite many decades of research, 
it is still an open problem. Recent approaches for traffic participants tracking are mostly image based, which suffer from motion 
blur and partial occlusion constantly. In this paper we  present a ConvNet architecture that can associate cameras as well as LIDAR 
data to produce accurate 3D trajectories. Towards this goal, a correlation module is introduced to capture object co-occurrences 
across time, and a multi-task objective for frame-based object detection and across-frame track regression is used, therefore, 
the network can performs detection and tracking simultaneously. Our proposed architecture is shown to produce competitive results 
on the KITTI Object tracking datasets. Code and models will be available soon.

#### ② Introduction



Object tracking is one of the most fundamental tasks in autonomous driving, it provides the diversity of location of different individual over time.  

Location, perception, planing and decision are key components in autonomous driving system, meanwhile, *perception* is the most significant part. 
Fed with location information captured by series of sensors, such as GPS, IMU, camera or Lidar, the task of perception module is to perceive s
urrounding environmental information and vehicle status, thus the following planning and decision subsystem can work well. 
While one of the most fundamental tasks in perception systems is the traffic participants tracking, which is classified as *Multi-target tracking*, 
....(simply describe the definition of MTT)

What is the critical problem in MTT? what is the critical problem in 2D object tracking? what is the mainly problem in 3D object tracking?

Two different approaches to MTT:

- tracking by filtering
- tracking by detection

what's the pros and cons of these two different approaches? which method is more compatible with 3D object tracking?
 what's the difficulty of utilizing "tracking by detection" approach in KITTI datasets.

In this paper, we present an end-to-end approach to settle 3D object tracking in auto driving scenery, 
which combine the RGB feature in image and accurate location information in point clouds by leveraging deep learning method. 
(simply describe the structure of Bi-AVOD)

Our contributions are n-fold: (1) 


#### ③ Related Work

a. Tracking

b. 3D object detection in Kitti

c. 3D object tracking

#### ④ Methodology

a. Description of Bi-AVOD, Need a graph to show the network structure.

 - Introduction of AVOD (mainly refer to [ku2018joint] )
 - Introduction of Correlation operation (mainly refer to [feichtenhofer2017detect])

b. Link detections to trajectories. Need Pseudo code to show the improved algorithm

​	- IOU-tracking method. (refer to [bochinski2017high] )

c. Evaluation Metrics of Tracking. A brief explanation of items included in KITTI tracking evaluation results. Need mathematical expression.

#### ⑤ Experiments and Results

a. Induction of KITTI Tracking Datasets 

b. Data preprocessing. Including *plane file*, *coordinate transformation*, *range determination*.

c. training. super-parameter: *lr, IoU threshold, batch-size, iterations*. Optimization method. GPU (Telsa P100), Memory requirements (8541MB)

d. Ablation Studies

- With/Without correlation layer. First need to evaluation the **effectiveness** of correlation. 

| Method   | MOTA | MOTP | MT   | ML   | IDS  | FRAG |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
|without Corr|0.00|0.00|0.00|0.00|0.00|0.00|
|with Corr|0.00|0.00|0.00|0.00|0.00|0.00|

- different correlation loss functions. L1, L2... (optional)
- 

e. different *max_displacement*, 5/10/20. acc. & time



f. different frame strides. 1/3/5... 



g. Comparison of tracking results in KITTI test set results with others.



h. different boxes association methods. (optional)


#### ⑥ Conclusions

