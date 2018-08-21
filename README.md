## How to Run

```bash
$ roscore
$ cd ~/catkin_ws
$ source devel/setup.bash
$ roslaunch json_prolog json_prolog.launch
$ cd ~/catkin_ws/src/knowrob/json_prolog/src/json_prolog 
$ python2 main.py
```

or just

```bash
$ cd scripts
$ bash run.sh
```

and then 
```bash
$ bash learn_reach.sh
```

on another terminal

## Requirements

- tensorflow==1.8.0
- gym
- ros
- knowrob
- python 2.7
- imageio
- pandas
- natsort
