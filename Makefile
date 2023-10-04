WEIGHTS_PATH := ../marl-thesis/models/asf/weights/SLOWFAST_EPIC.pyth
CONFIG_PATH := ../marl-thesis/models/asf/config/SLOWFAST_R50.yaml
INPUT_PATH := ../marl-thesis/data/EPIC-KITCHENS/P10/videos/P10_04_trimmed.wav

.PHONY: example
example:
	@python main.py \
		-w $(WEIGHTS_PATH) \
		-c $(CONFIG_PATH) \
		-f $(INPUT_PATH)

.PHONY: lint
lint:
	@ruff **/*.py --line-length 120 --fix
	@isort **/*.py --filter-files --profile black
	@black **/*.py -l 120