# pantomime-machine-learning

## How to run

1. Start `redis` server by running in the terminal:

```sh
sudo service redis-server start
```

2. Then run two workers in the separate terminals:

```sh
python worker.py streaming

python worker.py uploads
```

3. In the separate terminal, run `socketio` server:

```sh
python app.py
```
