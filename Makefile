PYTHON ?= python3

.PHONY: install test lint format app report

install:  ## Install dependencies and the rtm package
	$(PYTHON) -m pip install -r requirements.txt

test:  ## Run the unit tests
	$(PYTHON) -m pytest -q

lint:  ## Check style and formatting
	ruff check .
	ruff format --check .

format:  ## Apply formatting fixes
	ruff check --fix .
	ruff format .

app:  ## Launch the Streamlit dashboard
	$(PYTHON) -m streamlit run app.py

report:  ## Print the workings and rebuild figures, results and the one-page PDF
	$(PYTHON) -m rtm
	$(PYTHON) scripts/build_report.py
