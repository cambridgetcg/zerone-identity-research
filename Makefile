PYTHON ?= python3

.PHONY: check model prototype

check:
	$(PYTHON) -m unittest discover -s tests -v
	$(PYTHON) tools/check_repository.py
	$(PYTHON) tools/check_c2c_artifacts.py
	$(PYTHON) tools/check_checkpoint_branches.py
	$(PYTHON) tools/check_temporal_memory.py
	$(PYTHON) tools/check_identity_continuity.py
	$(PYTHON) prototypes/identity_changes/build.py --check

model:
	$(PYTHON) model/return_model.py

prototype:
	$(PYTHON) prototypes/identity_changes/build.py
