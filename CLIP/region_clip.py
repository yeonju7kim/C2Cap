# -*- encoding: utf-8 -*-
'''
@File    :   region_clip.py   
@Contact :   yeonju7.kim@gmail.com
@License :   (C)Copyright 2022-2024
 
@Modify Time      @Author        @Version    @Desciption
------------      -------        --------    -----------
2023-02-02 오전 7:38   yeonju7.kim      1.0         None
'''
import h5py
import numpy as np
import warnings

detections_path = '/mnt/hard1/yj_coco/MS_COCO_2014/annotations/coco_detections.hdf5'
def preprocess(x, avoid_precomp=False):
    image_id = int(x.split('_')[-1].split('.')[0])
    try:
        f = h5py.File(detections_path, 'r')
        # f['100000_boxes'][()](31,4), f['100000_cls_prob'][()](41,1601), f['100000_features'][()](31,2048)
        precomp_data = f['%d_features' % image_id][()]
        if sort_by_prob:
            precomp_data = precomp_data[np.argsort(np.max(f['%d_cls_prob' % image_id][()], -1))[::-1]]
        # precomp_data = self.att_loader.get(str(image_id))
    except KeyError:
        warnings.warn('Could not find detections for %d' % image_id)
        precomp_data = np.random.rand(10, 2048)

    delta = max_detections - precomp_data.shape[0]
    if delta > 0:
        precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
    elif delta < 0:
        precomp_data = precomp_data[:self.max_detections]

    return precomp_data.astype(np.float32)