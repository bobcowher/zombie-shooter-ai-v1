#!/bin/bash

RUNPOD_HOST=$1
RUNPOD_PORT=$2
RUNPOD_PROJECT_NAME=$3



ssh root@${RUNPOD_HOST} -p ${RUNPOD_PORT} "pgrep -f python | xargs kill -9"
ssh root@${RUNPOD_HOST} -p ${RUNPOD_PORT} "mkdir -p /app/${RUNPOD_PROJECT_NAME}"
scp -P ${RUNPOD_PORT} *.py root@${RUNPOD_HOST}:/app/${RUNPOD_PROJECT_NAME}
scp -r -P ${RUNPOD_PORT} images root@${RUNPOD_HOST}:/app/${RUNPOD_PROJECT_NAME}/images
scp -r -P ${RUNPOD_PORT} sounds root@${RUNPOD_HOST}:/app/${RUNPOD_PROJECT_NAME}/sounds
scp -P ${RUNPOD_PORT} requirements.txt root@${RUNPOD_HOST}:/app/${RUNPOD_PROJECT_NAME}
ssh root@${RUNPOD_HOST} -p ${RUNPOD_PORT} "pip install -r /app/${RUNPOD_PROJECT_NAME}/requirements.txt" 
ssh root@${RUNPOD_HOST} -p ${RUNPOD_PORT} "nohup python ./train.py &" 
ssh root@${RUNPOD_HOST} -p ${RUNPOD_PORT} "nohup tensorboard --logdir='runs' --port=6007" 


