#!/usr/bin/env python
# -*- coding: utf-8 -*-
import ctypes

import xgboost.core
import numpy as np
import redis
import shap


client = redis.Redis()

# https://github.com/dmlc/xgboost/issues/3013#issuecomment-355783364
def xgb_load_model(buf):
    if isinstance(buf, str):
        buf = buf.encode()
    bst = xgboost.core.Booster()
    n = len(buf)
    length = xgboost.core.c_bst_ulong(n)
    ptr = (ctypes.c_char * n).from_buffer_copy(buf)
    xgboost.core._check_call(
        xgboost.core._LIB.XGBoosterLoadModelFromBuffer(bst.handle, ptr, length)
    )  # segfault
    return bst


model = xgb_load_model(client.get("iris_xgb_v2"))

test = np.array([[5.1, 3.8, 1.5, 0.3], [5.7, 2.9, 4.2, 1.3]])

explainer = shap.Explainer(model, feature_names=["x1", "x2", "x3", "x4"])

for i in range(test.shape[0]):
    shap_values = explainer(np.array([test[i]]))
    shap.plots.waterfall(shap_values[0])
