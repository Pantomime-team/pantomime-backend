
from constants import classes
from loguru import logger
from typing import Tuple
from utils import resize_frame

import cv2
import numpy as np
import time
import torch


USE_CUDA = False

def init_model(model_path: str) -> None:

    global __model
    __model = torch.jit.load(model_path)
    __model.eval()

    if USE_CUDA and torch.cuda.is_available():
        __model = __model.cuda()


def rslrecognition_process_data(input_list, config, verbose: bool = False) -> str:

    input_tensor = np.stack(input_list, axis=1)[None][None]
    input_tensor = input_tensor.astype(np.float32)
    input_tensor = torch.from_numpy(input_tensor)

    if USE_CUDA and torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    if verbose:
        start_time = time.time()
        logger.info("Starting recognition")

    with torch.no_grad():
        output = __model(input_tensor)[0]

    gloss = str(classes[output.argmax().item()])

    if verbose:
        logger.info(
            f"Recognized \"{gloss}\" with confidence: {output.max().item() * 100:.2f}%"
        )
        logger.info(f"Recognition took {time.time() - start_time:.2f} seconds")
    if output.max() > config['threeshold']:
        return gloss
    return ""


def rslrecognition_stream(input_list, config, start_time, verbose: bool = False) -> Tuple[str, float]:

    fallback_seconds = time.time() - start_time
    logger.info(f"This job is {fallback_seconds:2f}s old")

    if len(input_list) != config["window_size"]:
        raise ValueError("Input list must be of length window_size")

    return rslrecognition_process_data(input_list, config, verbose), time.time() - start_time


def rslrecognition_video(path_to_input_video, config, remove_finish: bool = False, verbose: bool = False) -> str:

    cap = cv2.VideoCapture(path_to_input_video)

    input_list = []
    output_list = []

    frame_counter = 0

    while True:
        ret, frame = cap.read()

        if ret is False:
            break

        frame_counter += 1

        if frame_counter == config["frame_interval"]:
            image = process_frame(frame.copy(), config["mean"], config["std"])
            input_list.append(image)

            if len(input_list) == config["window_size"]:
                output = rslrecognition_process_data(
                    input_list, config, verbose
                )

                output_list.append(output)
                input_list = []

            frame_counter = 0

    cap.release()

    if remove_finish:
        import os
        os.remove(path_to_input_video)

    return " ".join(output_list)


def process_frame(frame, mean, std):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = resize_frame(frame)
    frame = (frame - mean) / std
    frame = np.transpose(frame, [2, 0, 1])

    return frame
