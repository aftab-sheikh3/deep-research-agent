FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[openai,anthropic,ui]"
COPY evals ./evals

RUN useradd --create-home app && mkdir -p /app/data && chown -R app /app
USER app
ENV CHECKPOINT_DB=/app/data/checkpoints.sqlite

EXPOSE 8000 8501
CMD ["uvicorn", "deep_research.api:app", "--host", "0.0.0.0", "--port", "8000"]
