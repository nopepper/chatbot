# Use an official Python runtime as a parent image
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

RUN apt update
RUN apt -y upgrade
RUN apt install -y build-essential libssl-dev libffi-dev python3-pip python3.10-venv

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install --user pipx
RUN python3 -m pipx ensurepath
RUN python3 -m pipx install poetry

ENV PATH="${PATH}:/root/.local/bin"

# Install any needed packages specified in requirements.txt
RUN poetry install --no-interaction --no-cache

# Make port 80 available to the world outside this container
EXPOSE 80

CMD ["sleep", "infinity"]