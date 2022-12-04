# yolov7_StrongSORT_ROS
Integration of  "Yolov7 StrongSort" with ROS for real time object tracking


# Requirements
Ubuntu 18.04

ROS Melodic


# Installation

Clone the repository recursively:

```
cd catkin_workspace/src
git clone --recurse-submodules https://github.com/PardisTaghavi/yolov7_StrongSORT_ROS.git
cd ../
catkin build yolov7_strongsort_ros or catkin_make
```

```
cd Yolov7_StrongSORT_OSNet
pip install -r requirements.txt
```

run through python code:
```
python trackRos.py 
```

or run through the launch file:
```
roslaunch yolov7_strongsort_ros trackingLaunch.launch
```

Name of the topic should be modified in the trackRos.py


# Speed
you can check frequency :
```
rostopic hz /trackResult
```
Currently it is around 7-8 Hz.
