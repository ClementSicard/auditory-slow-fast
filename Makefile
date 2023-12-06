.ONESHELL:

EK_REPO_NAME := epic-kitchens-100-annotations
ES_REPO_NAME := epic-sounds-annotations
EK_DL_NAME := epic-kitchens-download-scripts
EK_NAME := epic-kitchens-100
OUTPUT_DIR := output
DATA_DIR := data
LOGS_DIR := logs
VENV_DIR := slowfast
REPO_DIR := $${SCRATCH}/auditory-slow-fast
JOB_NAME := slowfast-training
MAIL_ADDRESS := $${USER}@nyu.edu

CONFIG_PATH := models/asf/config/SLOWFAST_R50.yaml

EXAMPLE_FILE := $(DATA_DIR)/EPIC-KITCHENS/P10/videos/P10_04_trimmed.wav

# Model weights
ASF_WEIGHTS_FILE := SLOWFAST_EPIC.pyth

# Conda activate
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


.PHONY: data
data: # This target clones the repos only if they don't exist in the data directory
	@mkdir -p $(DATA_DIR) # Create the data directory if it doesn't exist

	@if [ ! -d "$(DATA_DIR)/$(EK_REPO_NAME)" ]; then \
		cd $(DATA_DIR) && git submodule add https://github.com/epic-kitchens/$(EK_REPO_NAME) ; \
	fi

	@if [ ! -d "$(DATA_DIR)/$(ES_REPO_NAME)" ]; then \
		cd $(DATA_DIR) && git submodule add https://github.com/epic-kitchens/$(ES_REPO_NAME) ; \
	fi

	@if [ ! -d "$(DATA_DIR)/$(EK_DL_NAME)" ]; then \
		cd $(DATA_DIR) && git submodule add https://github.com/epic-kitchens/$(EK_DL_NAME) ; \
	fi

	$(MAKE) update


.PHONY: weights-asf
weights-asf:
	@mkdir -p models/asf/weights
	@wget https://www.dropbox.com/s/cr0c6xdaggc2wzz/$(ASF_WEIGHTS_FILE) -O models/asf/weights/$(ASF_WEIGHTS_FILE)

.PHONY: update
update:
	@git submodule sync --recursive
	@git submodule update --init --recursive
	@git pull --recurse-submodules


.PHONY: bash
bash:
	@echo "Running interactive bash session"
	@srun --job-name "interactive bash" \
		--cpus-per-task 4 \
		--mem 16G \
		--time 4:00:00 \
		--pty bash

.PHONY: bash-gpu
bash-gpu:
	@echo "Running interactive bash session"
	@srun --job-name "interactive bash" \
		--cpus-per-task 8 \
		--mem 16G \
		--gres gpu:1 \
		--time 4:00:00 \
		--pty bash

.PHONY: queue
queue:
	@squeue -u $(USER)

.PHONY: example-cluster
example-cluster:
	@python main.py \
		--model audio_slowfast \
		--config $(CONFIG_PATH) \
		--example $(EXAMPLE_FILE) \
		--verbs break crush pat shake sharpen smell throw water

.PHONY: example-local
example-local:
	@python main.py \
		--model audio_slowfast \
		--config config.local.yaml \
		--example $(EXAMPLE_FILE) \
		--make-plots \
		--verbs break crush pat shake sharpen smell throw water

.PHONY: example
example:
	@if echo "$(shell hostname)" | grep -q "nyu"; then \
		echo "Running on cluster"; \
		$(MAKE) example-cluster; \
	else \
		echo "Running locally"; \
		$(MAKE) example-local; \
	fi


.PHONY: lint
lint: # This target runs the formatter (black), linter (ruff) and sorts imports (isort)
	@isort . --skip $(DATA_DIR)/ --profile black
	@ruff . --fix --line-length 120 --show-source --exclude ./$(DATA_DIR) --force-exclude -v
	@black . --force-exclude ./$(DATA_DIR) --line-length 120 --color


.PHONY: test-code
test-code:
	@JUPYTER_PLATFORM_DIRS=1 pytest --ignore-glob $(DATA_DIR) --ignore-glob audio_slowfast --code-highlight yes -v

.PHONY: test-dataloader
test-dataloader:
	@JUPYTER_PLATFORM_DIRS=1 pytest --ignore-glob $(DATA_DIR) --ignore-glob audio_slowfast --code-highlight yes -v

.PHONY: update-deps
update-deps:
	@pip install -U -r requirements.txt

.PHONY: train
train:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	CUDA_LAUNCH_BLOCKING=1 python main.py \
		--model audio_slowfast \
		--config $(CONFIG_PATH) \
		--train \
		--verbs break crush pat shake sharpen smell throw water \
		--augment \
		--factor 4.0

.PHONY: train-small
train-small:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	python main.py \
		--model audio_slowfast \
		--config $(CONFIG_PATH) \
		--train \
		--verbs break crush pat shake sharpen smell throw water \
		--augment \
		--small

.PHONY: test
test:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	python main.py \
		--model audio_slowfast \
		--config $(CONFIG_PATH) \
		--test \
		--verbs break crush pat shake sharpen smell throw water \
		--augment \
		--factor 4.0


.PHONY: job-train
job-train:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-slowfast-train.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time 24:00:00 \
	    --mem 16G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name $(JOB_NAME) \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make train"

.PHONY: job-test
job-test:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d_%T"); \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-slowfast-train.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:v100:1 \
	    --time 12:00:00 \
	    --mem 16G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name $(JOB_NAME) \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make test"
