#!/bin/bash

echo "Running DVC pipeline..."

dvc repro --force

if [ $? -eq 0 ]; then
    echo "Pipeline successful. Pushing to remote..."
    dvc push
else
    echo "Pipeline failed.  Not pushing."
fi