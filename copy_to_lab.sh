#/bin/bash

DST_DIR=/workspace/bike_project/managed_bike/source/managed_bike/managed_bike
SRC_DIR=/workspace/lab_share/samples
cp -r $SRC_DIR/robots $DST_DIR/
cp -rf $SRC_DIR/mdp $DST_DIR/tasks/manager_based/managed_bike/
cp $SRC_DIR/managed_bike_env_cfg.py $DST_DIR/tasks/manager_based/managed_bike/
echo "done"
