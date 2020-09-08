# -*- coding: utf-8 -*-
# file: fine_tune_model.py
# author: JinTian
# time: 10/05/2017 9:54 AM
# Copyright 2017 JinTian. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
from torchvision import models
import torch
from torch import nn
import os
from global_config import MODEL_SAVE_FILE, USE_GPU, DEVICE


def fine_tune_model():

    model = models.mobilenet_v2(pretrained=True)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)

    if os.path.exists(MODEL_SAVE_FILE):
        print("loading saved model")
        if USE_GPU:
            model.load_state_dict(torch.load(MODEL_SAVE_FILE))
        else:
            model.load_state_dict(
                torch.load(MODEL_SAVE_FILE, map_location=lambda storage, loc: storage)
            )

    model = model.to(DEVICE)
    return model
