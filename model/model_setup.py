from time import perf_counter
from typing import Any, Callable
from copy import deepcopy

from huggingface_hub import hf_hub_download
import torch
import sys

def configure_model():
    """
    Subroutine to organize model setttings. 
    This will carry the characteristics of model which are based on another
    """

    model_id = "yermandy/GenD_CLIP_L_14"



def run_model(model, processed_input, predict_function):
    """
    Execution of model and prediction from data.
    Includes train and testing split and experiment environment.
    """
    prepared_data = processed_input["data"]

    start_time = perf_counter()

    score = predict_function(model, prepared_data)

    elapsed_time = perf_counter() - start_time

    return {
        "deepfake_score": score,
        "inference_seconds": elapsed_time,
        "metadata": processed_input["metadata"]
    }


