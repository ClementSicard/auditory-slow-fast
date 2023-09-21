WEIGHTS_PATH := ../thesis/models/asf/weights/SLOWFAST_EPIC.pyth
CONFIG_PATH := configs/EPIC-KITCHENS/SLOWFAST_R50.yaml
INPUT_PATH := ../thesis/data/EPIC-KITCHENS/P10/videos/P10_04_trimmed.wav
PREC_VOCAB := configs/preconditions.csv
POSTC_VOCAB := configs/postconditions.csv

.PHONY: example
example:
	@python auditory_slowfast.py \
		-w $(WEIGHTS_PATH) \
		-c $(CONFIG_PATH) \
		-f $(INPUT_PATH) \
		--prec_vocab $(PREC_VOCAB) \
		--postc_vocab $(POSTC_VOCAB) \