# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies for ImageMagick, Tesseract, Ghostscript, OpenCV, and Poppler
RUN apt-get update && apt-get install -y \
    imagemagick \
    tesseract-ocr \
    ghostscript \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN run: pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

# Install OpenCV
RUN pip install opencv-python

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV PYTHONUNBUFFERED=1

# Run the FastAPI application when the container launches
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
