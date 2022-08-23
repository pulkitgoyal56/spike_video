# Spike Video

Synchronises Electrophysiology Spikes to Video.  

## Dependencies

- [FFmpeg](https://ffmpeg.org/)  
- [ImageMagick](https://imagemagick.org/)  
- [Python3 requirements](./requirements.txt)  

## Setup

This package can be used using Docker to avoid tedious installations, especially on Windows.  

The included [*Dockerfile*](./Dockerfile) builds an image with all requirements installed. From the associated container, [a Jupyter notebook](./notebooks/application.ipynb) via Jupyter Lab at <http://localhost:8887/lab/tree/application.ipynb> is served which can give a sandboxed access to the package.  

1. Install Docker {[Linux](https://docs.docker.com/desktop/install/linux-install/), [Windows](https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe)}.  

2. Build Docker image.

   ``` sh
   docker build . --tag spike_video
   ```

3. Create and run Docker container.
   1. Linux

      ``` sh
      docker run -d -p 127.0.0.1:8887:8888 --name spike_video -v $(pwd)/data:/app/data -v $(pwd)/output:/app/.temp -v $(pwd)/notebooks:/app/notebooks spike_video
      ```

   2. Windows (PowerShell)

      ``` powershell
      docker run -d -p 127.0.0.1:8887:8888 --name spike_video -v ${pwd}/data:/app/data -v ${pwd}/output:/app/.temp -v ${pwd}/notebooks:/app/notebooks spike_video
      ```

   > Note: The names of folders used for output are different in the host and the Docker container/application.  
   > |  Host  | Container |  
   > |--------|-----------|  
   > | output |   .temp   |  

4. Go to <http://localhost:8887/lab/tree/application.ipynb>.  

The script [setup.ps1](./setup.ps1)/[setup.sh](./setup.sh) provides commands 2, 3, and 4 above. Run it with PowerShell/'sh' respectively.  

## Running/Stopping

> Start

``` sh
docker start spike_video
```

> Stop

``` sh
docker stop spike_video
```
