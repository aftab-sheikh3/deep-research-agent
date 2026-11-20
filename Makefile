.PHONY: install test lint eval api ui docker

install:
	pip install -e ".[dev,ui]"

test:
	pytest -q

lint:
	ruff check src tests evals

eval:            ## real model + real search, e.g. LLM_PROVIDER=openai make eval
	python -m evals.run_eval

api:
	uvicorn deep_research.api:app --reload --port 8000

ui:
	streamlit run src/deep_research/ui.py

docker:
	docker compose up --build
