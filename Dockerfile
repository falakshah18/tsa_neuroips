FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from spikingjelly.activation_based import neuron; print('spikingjelly OK')"
RUN python -c "import tonic; print('tonic OK')"

# Sanity check
RUN python -m pytest tests/ -q

ENTRYPOINT ["python", "main.py"]
