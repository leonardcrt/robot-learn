# Image d'entraînement / évaluation (CPU).
#   docker build -f docker/Dockerfile -t robot-learn .
#   docker run --rm -v "$PWD/results:/robot-learn/results" robot-learn                     # benchmark
#   docker run --rm -v "$PWD/checkpoints:/robot-learn/checkpoints" robot-learn python -m rl.train
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
WORKDIR /robot-learn

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml requirements.txt README.md ./
COPY env ./env
COPY expert ./expert
COPY models ./models
COPY bc ./bc
COPY rl ./rl
COPY eval ./eval
RUN pip install -e ".[dev]"

COPY tests ./tests
COPY checkpoints ./checkpoints

CMD ["python", "-m", "eval.benchmark"]
