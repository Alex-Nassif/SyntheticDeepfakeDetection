from time import perf_counter
from typing import Any, Callable
from copy import deepcopy

from huggingface_hub import hf_hub_download
import torch
import sys
import importlib.util

def configure_model():
    """
    Subroutine to organize model setttings. 
    This will carry the characteristics of model which are based on another
    """

    model_id = "yermandy/GenD_CLIP_L_14"

    model_file = hf_hub_download(
        repo_id=model_id,
        filename="modeling_gend.py",
    )

    spec = importlib.util.spec_from_file_location("gend_hf_model", model_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    model = module.GenD.from_pretrained(
        model_id,
        low_cpu_mem_usage=False
    )    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #always be cpu for standard case
    model = model.to(device)
    model.eval()

    return model

def run_model(model, processed_input, predict_function):
    """
    Execution of model and prediction from data.
    Includes train and testing split and experiment environment.
    """
    prepared_data = processed_input["data"]

    device = next(model.parameters()).device
    prepared_data = prepared_data.to(device)

    model.eval()

    start_time = perf_counter()

    with torch.inference_mode():
        logits = model(prepared_data)
        if logits.shape != (1, 2):
            raise ValueError("Expected two classification outputs.")
        probabilities = logits.softmax(dim=-1)

    score = probabilities[0, 1].item()

    elapsed_time = perf_counter() - start_time

    return {
        "deepfake_score": score,
        "inference_seconds": elapsed_time,
        "metadata": processed_input["metadata"]
    }


