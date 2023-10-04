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
	@ruff . --line-length 120 --fix
	@isort **/*.py --filter-files --profile black
	@black . -l 120

.PHONY: bash
bash:
	@echo "Running interactive bash session"
	@srun --job-name "interactive bash" \
		--cpus-per-task 8 \
		--mem 16G \
		--time 12:00:00 \
		--pty bash