# Akida engine 2.5.0 is what the BB15 library loads (src/akida/version.h).
# akida 2.5.0 ships Linux/Windows wheels only, hence the container.
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install --timeout 120 --retries 10 -r /tmp/requirements.txt
# akida-models 1.2.0 was current when MetaTF 2.5.0 shipped; its pins point at newer
# releases, so it is installed without them, only for the pretrained KWS model.
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps akida-models==1.2.0
WORKDIR /work
