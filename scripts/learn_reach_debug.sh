source ~/catkin_ws/devel/setup.bash

python2 -m pdb main.py --metatrain_iterations=30000 --meta_batch_size=5 --init=xavier --train_update_lr=0.01 \
    --clip=True --clip_min=-10 --clip_max=10 --no_action=True --fp=True --num_filters=40 --filter_size=3 --num_conv_layers=4 --num_strides=4 \
    --fc_bt=True --all_fc_bt=False --bt_dim=20 --two_head=True --temporal_conv_2_head=True \
    --gpu_memory_fraction=1 --train=True --resume=False
