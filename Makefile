WEIGHTS_PATH := ../marl-thesis/models/asf/weights/SLOWFAST_EPIC.pyth
CONFIG_PATH := ../marl-thesis/models/asf/config/SLOWFAST_R50.yaml
INPUT_PATH := ../marl-thesis/data/EPIC-KITCHENS/P10/videos/P10_04_trimmed.wav

.PHONY: example
example:
	@python main.py \
		-w $(WEIGHTS_PATH) \
		-c $(CONFIG_PATH) \
		-f $(INPUT_PATH)

