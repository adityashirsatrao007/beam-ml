FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code

# Run gunicorn on port 7860 with long timeout for model load
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
