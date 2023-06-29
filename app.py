from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from model import rslrecognition, process_frame
from redis import Redis
from rq import Queue
from rq.job import JobStatus
from utils import readb64

import os
import toml


if __name__ == "__main__":
    # Read toml config file
    if not os.path.exists("config.toml"):
        raise FileNotFoundError("config.toml not found.")

    with open("config.toml", "r") as f:
        config = toml.load(f)

    # Load config data
    model_path = config["model_path"]
    model_mean = config["mean"]
    model_std = config["std"]
    # frame_interval = config["frame_interval"]
    window_size = config["window_size"]

    # Setup Redis Queue
    q = Queue(connection=Redis())

    # Setup Flask and SocketIO
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "secret!"
    socketio = SocketIO(app)

    rlss_sessions = {}  # { sid: { "input_list": [], "jobs_list": [] } }

    @app.route("/")
    def index():
        return render_template("index.html")

    def drop_session(sid):
        if sid in rlss_sessions:
            # Cancel all jobs
            for job in rlss_sessions[sid]["jobs_list"]:
                # job.cancel()
                job.delete()

            del rlss_sessions[sid]

    @socketio.on("connect")
    def handle_connect():
        drop_session(request.sid)

        rlss_sessions[request.sid] = {
            "input_list": [],
            "jobs_list": []
        }

    @socketio.on("disconnect")
    def handle_disconnect():
        drop_session(request.sid)

    @socketio.on("data_stream")
    def handle_datastream(data):
        if request.sid not in rlss_sessions:
            return

        # TODO: check timestamps to prevent ddos

        rlss_sess = rlss_sessions[request.sid]

        # Base64 -> frame
        frame = process_frame(readb64(data), model_mean, model_std)
        rlss_sess["input_list"].append(frame)

        if len(rlss_sess["input_list"]) > window_size:
            # TODO: frame dropping, if overloaded
            input_tensor = rlss_sess["input_list"][:window_size]
            del rlss_sess["input_list"][:window_size]

            job = q.enqueue(
                rslrecognition, model_path, input_tensor, config, True
            )
            rlss_sess["jobs_list"].append(job)

        # Polling for the results
        finished_jobs = []
        for job in rlss_sess["jobs_list"]:
            match job.get_status():
                case JobStatus.FINISHED:
                    emit("data_recognised", {"gloss": job.return_value()})
                    finished_jobs.append(job)
                case JobStatus.FAILED:
                    finished_jobs.append(job)
                case _:
                    pass  # TODO: handle other cases

        # Remove finished jobs
        for job in finished_jobs:
            rlss_sess["jobs_list"].remove(job)

    # Run server
    socketio.run(app, port=8000)
