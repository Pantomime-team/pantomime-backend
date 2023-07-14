#! /bin/bash
sudo service redis-server start

redis-cli FLUSHDB
redis-cli FLUSHALL

pkill -9 -f rq
while pgrep rq > /dev/null; do sleep 1; done

rq worker streaming &
rq worker uploads &

python app.py

killall -q rq
while pgrep rq > /dev/null; do sleep 1; done
