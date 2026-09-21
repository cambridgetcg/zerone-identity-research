PYTHON ?= python3

.PHONY: check model

check:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) tools/check_repository.py
	$(PYTHON) tools/check_c2c_artifacts.py

model:
	$(PYTHON) model/return_model.py
