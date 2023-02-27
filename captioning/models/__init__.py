from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy

import numpy as np
import torch

from .ShowTellModel import ShowTellModel
from .FCModel import FCModel
from .AttModel import *
from .TransformerModel import TransformerModel
from .TransformerModel_c3cap import TransformerModel_c3cap
from .TransformerModel_c3cap_no_projection_decoder import TransformerModel_c3cap_no_projection_decoder
from .TransformerModel_c3cap_no_projection import TransformerModel_c3cap_no_projection
from .TransformerModel_c3cap_no_projection_instance_normalize import \
    TransformerModel_c3cap_no_projection_instance_normalize
from .TransformerModel_c3cap_no_projection_same_structure import TransformerModel_c3cap_no_projection_same_structure
from .TransformerModel_c3cap_scaled import TransformerModel_c3cap_scaled
from .TransformerModel_c3cap_l2distance import TransformerModel_c3cap_l2distance
from .TransformerModel_c3cap_l2distance_scaled import TransformerModel_c3cap_l2distance_scaled
from .TransformerModel_c3cap_att_fuse import TransformerModel_c3cap_att_fuse
from .cachedTransformer import TransformerModel as cachedTransformer
from .BertCapModel import BertCapModel
from .M2Transformer import M2TransformerModel
from .AoAModel import AoAModel

def setup(opt):
    if opt.caption_model in ['fc', 'show_tell']:
        print('Warning: %s model is mostly deprecated; many new features are not supported.' %opt.caption_model)
        if opt.caption_model == 'fc':
            print('Use newfc instead of fc')
    if opt.caption_model == 'fc':
        model = FCModel(opt)
    elif opt.caption_model == 'language_model':
        model = LMModel(opt)
    elif opt.caption_model == 'newfc':
        model = NewFCModel(opt)
    elif opt.caption_model == 'show_tell':
        model = ShowTellModel(opt)
    # Att2in model in self-critical
    elif opt.caption_model == 'att2in':
        model = Att2inModel(opt)
    # Att2in model with two-layer MLP img embedding and word embedding
    elif opt.caption_model == 'att2in2':
        model = Att2in2Model(opt)
    elif opt.caption_model == 'att2all2':
        print('Warning: this is not a correct implementation of the att2all model in the original paper.')
        model = Att2all2Model(opt)
    # Adaptive Attention model from Knowing when to look
    elif opt.caption_model == 'adaatt':
        model = AdaAttModel(opt)
    # Adaptive Attention with maxout lstm
    elif opt.caption_model == 'adaattmo':
        model = AdaAttMOModel(opt)
    # Top-down attention model
    elif opt.caption_model in ['topdown', 'updown']:
        model = UpDownModel(opt)
    # StackAtt
    elif opt.caption_model == 'stackatt':
        model = StackAttModel(opt)
    # DenseAtt
    elif opt.caption_model == 'denseatt':
        model = DenseAttModel(opt)
    # Transformer
    elif opt.caption_model == 'transformer':
        if getattr(opt, 'cached_transformer', False):
            model = cachedTransformer(opt)
        else:
            model = TransformerModel(opt)
    elif opt.caption_model == 'transformer_c3cap':
        if opt.scaled:
            model = TransformerModel_c3cap_scaled(opt)
        else:
            model = TransformerModel_c3cap(opt)
    elif opt.caption_model == 'transformer_c3cap_l2distance':
        if opt.scaled:
            model = TransformerModel_c3cap_l2distance_scaled(opt)
        else:
            model = TransformerModel_c3cap_l2distance(opt)
    elif opt.caption_model == 'transformer_c3cap_att_fuse':
        model = TransformerModel_c3cap_att_fuse(opt)
    elif opt.caption_model == 'transformer_c3cap_no_projection_decoder':
        model = TransformerModel_c3cap_no_projection_decoder(opt)
    elif opt.caption_model == 'transformer_c3cap_no_projection':
        model = TransformerModel_c3cap_no_projection(opt)
    elif opt.caption_model == 'transformer_c3cap_no_projection_instance_normalize':
        model = TransformerModel_c3cap_no_projection_instance_normalize(opt)
    elif opt.caption_model == 'transformer_c3cap_no_projection_same_structure':
        model = TransformerModel_c3cap_no_projection_same_structure(opt)
    # AoANet
    elif opt.caption_model == 'aoa':
        model = AoAModel(opt)
    elif opt.caption_model == 'bert':
        model = BertCapModel(opt)
    elif opt.caption_model == 'm2transformer':
        model = M2TransformerModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    return model
