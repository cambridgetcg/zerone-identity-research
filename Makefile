PYTHON ?= python3

.PHONY: check model

check:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) tools/check_repository.py

model:
	$(PYTHON) model/return_model.py
