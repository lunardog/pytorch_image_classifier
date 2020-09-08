# -*- coding: utf-8 -*-
# file: predict.py
# author: JinTian
# time: 10/05/2017 9:52 AM
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
"""
this file predict single image using the model we previous trained.
"""
from models.fine_tune_model import fine_tune_model
from global_config import IMAGE_SIZE, MODEL_SAVE_FILE, DEVICE
import torch
import os
import sys
from data_loader.data_loader import DataLoader


def predict_single_image(inputs, classes_name):
    model = fine_tune_model()
    inputs = inputs.to(DEVICE)
    if not os.path.exists(MODEL_SAVE_FILE):
        print("can not find model save file.")
        exit()
    else:
        model.load_state_dict(torch.load(MODEL_SAVE_FILE))
        outputs = model(inputs)

        print(outputs.data)
        prediction_tensor = torch.argmax(outputs)
        _, predict = torch.min(outputs.data, 1)
        print(predict)

        prediction = prediction_tensor.numpy()
        print(prediction)
        print("predict: ", prediction)
        print("this is {}".format(classes_name[prediction]))


def predict():
    if len(sys.argv) > 1:
        print("predict image from : {}".format(sys.argv[1]))
        data_loader = DataLoader(
            data_dir="datasets/hymenoptera_data", image_size=IMAGE_SIZE
        )
        if os.path.exists(sys.argv[1]):
            inputs = data_loader.make_predict_inputs(sys.argv[1])
            predict_single_image(inputs, data_loader.data_classes)
    else:
        print("must specific image file path.")


if __name__ == "__main__":
    predict()
