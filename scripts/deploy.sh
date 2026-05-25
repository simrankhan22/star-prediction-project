#!/bin/bash

echo "Deploying project to production server..."

rsync -avz ~/star-predictor/ ubuntu@130.238.29.82:/home/ubuntu/star-predictor/

echo "Deployment complete!"
