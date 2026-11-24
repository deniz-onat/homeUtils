# Akida engine 2.5.0 is what the BB15 library loads (src/akida/version.h).
# akida 2.5.0 ships Linux/Windows wheels only, hence the container.
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r /tmp/requirements.txt
WORKDIR /work
