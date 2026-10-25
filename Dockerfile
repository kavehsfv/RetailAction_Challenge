FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    git-lfs \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/RetailAction_WACV2027_Submission

COPY requirements.txt /workspace/RetailAction_WACV2027_Submission/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /workspace/RetailAction_WACV2027_Submission/

# Default command reproduces submission.csv from mounted data/raw and weights/
CMD ["python", "reproduce_submission.py", "--data-root", "data/raw", "--out", "submission.csv"]
