.PHONY: install test lint check notebooks dashboard compare api
install:
	python -m pip install -r requirements-dev.txt
test:
	python -m pytest -q
lint:
	ruff check src tests examples apps api scripts
notebooks:
	python scripts/check_notebooks.py
check: lint test notebooks
dashboard:
	streamlit run apps/portfolio_dashboard.py
compare:
	python -m src.cli compare --dataset iris
api:
	uvicorn api.main:app --reload
