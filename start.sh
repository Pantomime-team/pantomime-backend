#! /bin/bash
sudo service redis-server start

redis-cli FLUSHDB
redis-cli FLUSHALL

# pkill -9 -f rq
# while pgrep rq > /dev/null; do sleep 1; done

python worker.py streaming &
python worker.py uploads &

python app.py

# killall -q rq
# while pgrep rq > /dev/null; do sleep 1; done
