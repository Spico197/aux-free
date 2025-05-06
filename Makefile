all: format clean pre test
	echo 'finished'

.PHONY: format
format:
	ruff check --select I --fix .
	ruff format .

.PHONY: test
test:
	coverage run --source src -m pytest -vv .
	coverage report -m

.PHONY: pre
pre:
	pre-commit run --all-files

.PHONY: debug
debug:
	pytest -vv tests/data/test_tic_tac_toe.py

.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -f .coverage
	rm -f coverage.xml
	find . | grep -E '(__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf
