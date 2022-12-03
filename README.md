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
catkin build [pkg name] or catkin_make
```
```

cd ***
pip install -r requirements.txt
```

run through python code:
```
python track.py 
```
or run through the launch file:
```
roslaunch [pkg_name] [launchfile]
```
Name of the topic with data fromat /Image/sensor_msgs should be modified in the launch file
