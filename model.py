from constants import classes
from loguru import logger
from utils import resize_frame

import cv2
import numpy as np
import torch


def rslrecognition(model_path, input_list, config, verbose: bool = False) -> str:
    
    model = torch.jit.load(model_path)
    model.eval()

    if len(input_list) != config["window_size"]:
        raise ValueError("Input list must be of length window_size")

    input_tensor = np.stack(input_list, axis=1)[None][None]
    input_tensor = input_tensor.astype(np.float32)
    input_tensor = torch.from_numpy(input_tensor)

    with torch.no_grad():
        output = model(input_tensor)[0]

    gloss = str(classes[output.argmax().item()])

    if verbose:
        logger.info(
            f"Recognized \"{gloss}\" with confidence: {output.max().item() * 100:.2f}%"
        )

    return gloss


def process_frame(frame, mean, std):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = resize_frame(frame)
    frame = (frame - mean) / std
    frame = np.transpose(frame, [2, 0, 1])

    return frame
