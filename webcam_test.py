from model import rslrecognition, process_frame
from redis import Redis
from rq import Queue
from rq.job import JobStatus

import cv2
import os
import toml


if __name__ == "__main__":
    # Read yaml config file
    if not os.path.exists("config.toml"):
        raise FileNotFoundError("config.toml not found.")

    with open("config.toml", "r") as f:
        config = toml.load(f)

    # Load config data
    model_path = config["model_path"]
    model_mean = config["mean"]
    model_std = config["std"]
    frame_interval = config["frame_interval"]
    frame_interval_ticker = 0
    window_size = config["window_size"]

    # Test from video
    input_list = []

    q = Queue(connection=Redis())
    jobs_list = []

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        if frame_interval_ticker % frame_interval == 0:
            processed_frame = process_frame(
                frame, model_mean, model_std
            )

            input_list.append(processed_frame)
            frame_interval_ticker = 0

        if len(input_list) > window_size:
            input_tensor = input_list[:window_size]
            del input_list[:window_size]

            job = q.enqueue(
                rslrecognition, model_path, input_tensor, config, True
            )
            jobs_list.append(job)

        # Polling for finished jobs
        finished_jobs = []
        for job in jobs_list:
            match job.get_status():
                case JobStatus.FINISHED:
                    print(job.return_value())
                    finished_jobs.append(job)
                case JobStatus.FAILED:
                    finished_jobs.append(job)
                case _:
                    pass

        for job in finished_jobs:
            jobs_list.remove(job)

        frame_interval_ticker += 1

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
