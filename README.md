# kitti360-map-python
kitti360 global Lidar map making and sub-maps cutting in python

![image](https://github.com/Zhaozhpe/kitti360-global-map-python/blob/master/IMG/result.png "Sequence 00")

This code is originated from on [preprocess/kitti_maps.py](https://github.com/cattaneod/CMRNet/blob/master/preprocess/kitti_maps.py), but adds the following supports.
- [x] Global Map Making for [KITTI360](https://www.cvlibs.net/datasets/kitti-360/)
- [x] Sub-maps saved in `.bin` and `.pcd` (optional)
- [x] Remove ground of Lidar map
