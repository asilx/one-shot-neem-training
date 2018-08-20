#!/bin/bash

FNAME=${1:-learn_reach.sh}
CWD=$(pwd)
echo $FNAME
# Assume that you are running roscore on background

cd ~/catkin_ws
source devel/setup.bash
roslaunch json_prolog json_prolog.launch

# open another terminal and run the learning or testing process
xterm -e $CWD/$FNAME
