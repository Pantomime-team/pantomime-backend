# pantomime-machine-learning

## How to run

1. Start `redis` server by running in the terminal:

```sh
sudo service redis-server start
```

2. Then run:

```sh
rq worker
```

3. In the separate terminal, run `socketio` server:

```sh
python .\app.py
```
