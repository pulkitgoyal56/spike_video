# Script to setup Docker container

# Build Docker Image
docker build . --tag spike_video

# Create and Run Docker Container
docker run -d -p 127.0.0.1:8887:8888 --name spike_video -v ${pwd}/data:/app/data -v ${pwd}/output:/app/.temp -v ${pwd}/notebooks:/app/notebooks spike_video

# Open Jupyter Lab
Start-Process http://localhost:8887/lab/tree/application.ipynb