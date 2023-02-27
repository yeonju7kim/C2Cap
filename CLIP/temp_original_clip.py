# -*- encoding: utf-8 -*-
'''
@File    :   temp_original_clip.py   
@Contact :   yeonju7.kim@gmail.com
@License :   (C)Copyright 2022-2024
 
@Modify Time      @Author        @Version    @Desciption
------------      -------        --------    -----------
2023-02-15 오전 10:34   yeonju7.kim      1.0         None
'''
arg_file = 'clip_feature_train_aug3_all_image.hdf5'
ori_file = 'clip_feature_all_image.hdf5'

import h5py

f = h5py.File(arg_file, 'r')
ori_f = h5py.File(ori_file, 'w')
for id in f.keys():
    new_group = ori_f.create_group(id)
    new_group.create_dataset('image', data=f[str(id)]['image'][3])
f.close()
ori_f.close()