#! /bin/bash
redis-cli FLUSHDB
redis-cli FLUSHALL

pkill -f rq
pkill -f rq
pkill -f rq

rq worker streaming &
rq worker uploads &

python app2.py
