

from model_jobs import init_model, rslrecognition_stream, rslrecognition_process_data, rslrecognition_video
from rq import Connection, SimpleWorker

import os
import sys
import toml


with Connection():
    qs = sys.argv[1:] or ['default']

    # Read toml config file
    if not os.path.exists("config.toml"):
        raise FileNotFoundError("config.toml not found.")

    with open("config.toml", "r") as f:
        config = toml.load(f)

    # Preload model
    init_model(config["model_path"])

    w = SimpleWorker(qs)
    w.work()
