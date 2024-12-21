#!/bin/bash

RUNPOD_HOST=$1
RUNPOD_PORT=$2
RUNPOD_PROJECT_NAME=$3

mkdir runpod_models
scp -P ${RUNPOD_PORT} root@${RUNPOD_HOST}:/app/${RUNPOD_PROJECT_NAME}/models/* ./runpod_models/
