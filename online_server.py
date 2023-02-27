# -*- encoding: utf-8 -*-
'''
@File    :   online_server.py   
@Contact :   yeonju7.kim@gmail.com
@License :   (C)Copyright 2022-2024
 
@Modify Time      @Author        @Version    @Desciption
------------      -------        --------    -----------
2023-02-20 오후 10:34   yeonju7.kim      1.0         None
'''
import json
import sys
import torch

train = torch.load('eval_results/saved_pred_scst_no_projection_pretrained_b30_230216_train.pth')
val = torch.load('eval_results/saved_pred_scst_no_projection_pretrained_b30_230216_val.pth')
test = torch.load('eval_results/saved_pred_scst_no_projection_pretrained_b30_230216_test.pth')
# test = torch.load(open('scst_no_projection_pretrained_b30_230216_test.json','rb'))
# val = json.load(open('scst_no_projection_pretrained_b30_230216_val.json','rb'))

ans = []
for t in test:
    ans.append({"image_id":t["image_id"], "caption":t["caption"]})
json.dump(ans, open('captions_test2014_c2cap_results.json', 'w'))
ans = []
for v in val:
    ans.append({"image_id":v["image_id"], "caption":v["caption"]})
json.dump(ans, open('captions_val2014_c2cap_results.json', 'w'))

sys.path.append("/home/kyj/projects/c3cap-ver2/coco-caption")
from pycocotools.coco import COCO
cocoGt=COCO('captions_val2014.json')
cocoDt=cocoGt.loadRes('captions_test2014_c2cap_results.json')
a=1