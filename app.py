
from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit
from model_jobs import process_frame, rslrecognition_stream, rslrecognition_video
from redis import Redis
from rq import Queue
from rq.job import JobStatus
from time import time
from utils import readb64

import os
import toml


MAX_HTTP_BUFFER_SIZE = 1e8  # 100 MB

QUEUE_STREAMING = "streaming"
QUEUE_UPLOADS = "uploads"

KILL_CONNECTION_TIMEOUT_SECONDS = 10

SERVER_PORT = 5000


if __name__ == "__main__":

    # Read toml config file
    if not os.path.exists("config.toml"):
        raise FileNotFoundError("config.toml not found.")

    with open("config.toml", "r") as f:
        config = toml.load(f)

    # Load config data
    model_mean = config["mean"]
    model_std = config["std"]
    window_size = config["window_size"]

    # Setup Redis Queues
    q_streaming = Queue(QUEUE_STREAMING, connection=Redis())
    q_uploads = Queue(QUEUE_UPLOADS, connection=Redis(), default_timeout=600)

    # Setup session dict
    rlss_sessions = {}

    # request.sid: {
    #   "last_active": timestamp
    #   "streaming": {
    #       "frames_buffer": [], # TODO: user heapq instead of list
    #       "jobs_list": []
    #   },
    #   "uploads": {
    #       "file_buffer": [],
    #       "jobs_list": []
    #   }

    # Setup Flask and SocketIO
    flask_app = Flask(
        __name__,
        static_url_path="/",
        static_folder="static"
    )
    socketio = SocketIO(
        flask_app,
        max_http_buffer_size=MAX_HTTP_BUFFER_SIZE
    )

    # Setup routes
    @flask_app.route("/")
    def index():
        return send_from_directory("static", "index.html")

    def create_session(sid):
        rlss_sessions[sid] = {
            "last_active": time(),
            "streaming": {
                "frames_buffer": [],
                "jobs_list": []
            },
            "uploads": {
                "file_buffer": None,
                "jobs_list": []
            }
        }

    def drop_session(sid):
        if sid not in rlss_sessions:
            return

        # Cancel all jobs
        for job in rlss_sessions[sid]["streaming"]["jobs_list"]:
            job.cancel()
            job.delete()

        for job in rlss_sessions[sid]["uploads"]["jobs_list"]:
            job.cancel()
            job.delete()

        # Delete session
        del rlss_sessions[sid]

    def get_session(sid, update_last_active=True):
        if sid not in rlss_sessions:
            return None

        if update_last_active:
            rlss_sessions[sid]["last_active"] = time()

        return rlss_sessions[sid]

    @socketio.on("connect")
    def on_connect():
        # Remove session if it exists
        drop_session(request.sid)

        # Create new session
        create_session(request.sid)

    @socketio.on("disconnect")
    def on_disconnect():
        drop_session(request.sid)

    @socketio.on("webcam-streaming")
    def on_webcam_streaming(data):
        rlss_session = get_session(request.sid)
        if rlss_session is None:
            create_session(request.sid)

        rlss_sess_streaming = rlss_session["streaming"]

        image = data["image"]

        # Base64 -> frame
        frame = process_frame(readb64(image), model_mean, model_std)
        rlss_sess_streaming["frames_buffer"].append(frame)

        if len(rlss_sess_streaming["frames_buffer"]) > window_size:
            # TODO: frame dropping, if overloaded
            input_tensor = rlss_sess_streaming["frames_buffer"][:window_size]
            del rlss_sess_streaming["frames_buffer"][:window_size]

            job = q_streaming.enqueue(
                rslrecognition_stream,
                input_tensor,
                config,
                start_time=time(),
                verbose=True
            )
            rlss_sess_streaming["jobs_list"].append(job)

        # Polling for the results
        # expired_jobs = []
        finished_jobs = []

        for job in rlss_sess_streaming["jobs_list"]:
            match job.get_status():
                case JobStatus.FINISHED:
                    gloss, turnaround_time = job.return_value()

                    emit("recognized-caption", {
                        "gloss": gloss,
                        "turnaround_time": turnaround_time
                    })

                    finished_jobs.append(job)
                case JobStatus.FAILED | JobStatus.CANCELED:
                    finished_jobs.append(job)
                case _:
                    pass  # TODO: handle other cases

        # Remove finished jobs
        for job in finished_jobs:
            rlss_sess_streaming["jobs_list"].remove(job)

    @socketio.on("video-upload")
    def on_video_upload(data):
        rlss_session = get_session(request.sid)
        if rlss_session is None:
            create_session(request.sid)

        rlss_sess_upload = rlss_session["uploads"]

        filename = data["filename"]
        data_chunk = data["data_chunk"]
        current_chunk = data["current_chunk"]
        total_chunks = data["total_chunks"]

        if rlss_sess_upload["file_buffer"] is None:
            rlss_sess_upload["file_buffer"] = [None] * total_chunks

        rlss_sess_upload["file_buffer"][current_chunk] = data_chunk

        requested_chunk = None
        for i, buffer_chunk in enumerate(rlss_sess_upload["file_buffer"]):
            if buffer_chunk is None:
                requested_chunk = i
                break

        if requested_chunk is None:
            # All chunks have been received
            hashed_filename = str(hash(filename + str(time())))

            # Save video to disk
            with open(f"/tmp/{hashed_filename}.mp4", "wb") as f:
                for chunk in rlss_sess_upload["file_buffer"]:
                    f.write(chunk)

            process_video(f"/tmp/{hashed_filename}.mp4", request.sid)

            # Empty file buffer
            rlss_sess_upload["file_buffer"] = None

        emit("on-video-upload-status", {
            "status": "finished" if requested_chunk is None else "uploading",
            "details": None,
            "return_value": requested_chunk
        })

    def process_video(video_path, sid):
        rlss_session = get_session(sid)
        if rlss_session is None:
            create_session(request.sid)

        if len(rlss_session["uploads"]["jobs_list"]) > 0:
            # TODO: return error to client. Only one video can be processed at a time.
            return

        job = q_uploads.enqueue(
            rslrecognition_video,
            video_path,
            config,
            remove_finish=True,
            verbose=True
        )

        rlss_session["uploads"]["jobs_list"].append(job)

    @socketio.on("get-video-process-status")
    def on_get_video_process_status():
        rlss_session = get_session(request.sid)
        if rlss_session is None:
            emit("get-video-process-status", {
                "status": JobStatus.FAILED,
                "details": "Session not found.",
                "return_value": None
            })
            return

        if len(rlss_session["uploads"]["jobs_list"]) == 0:
            emit("get-video-process-status", {
                "status": JobStatus.FAILED,
                "details": "No video is being processed.",
                "return_value": None
            })
            return

        job = rlss_session["uploads"]["jobs_list"][0]
        match job.get_status():
            case JobStatus.FINISHED:
                job_result = job.return_value()

                # Remove finished jobs
                rlss_session["uploads"]["jobs_list"].clear()
            case _:
                job_result = None

        emit("get-video-process-status", {
            "status": job.get_status(),
            "details": None,
            "return_value": job_result
        })

    # Run server
    socketio.run(flask_app, debug=True, port=SERVER_PORT)
