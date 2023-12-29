.ONESHELL:

EK_REPO_NAME := epic-kitchens-100-annotations
ES_REPO_NAME := epic-sounds-annotations
EK_DL_NAME := epic-kitchens-download-scripts
EK_NAME := epic-kitchens-100
OUTPUT_DIR := output
DATA_DIR := data
LOGS_DIR := logs
VENV_DIR := slowfast
SCRATCH := /scratch/cs7561
REPO_DIR := $(SCRATCH)/auditory-slow-fast
JOB_NAME := asf-gru
MAIL_ADDRESS := $${USER}@nyu.edu
DURATION := 48:00:00
WANDB_CACHE_DIR := $(SCRATCH)/.cache/wandb
WANDB_DATA_DIR := $(SCRATCH)/.cache/wandb/data

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


.PHONY: weights-asf-vgg
weights-asf-vgg:
	@mkdir -p models/asf/weights
	@wget https://www.dropbox.com/s/oexan0vv01eqy0k/SLOWFAST_VGG.pyth -O models/asf/weights/SLOWFAST_VGG.pyth

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
	@srun --job-name "bash" \
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


.PHONY: train-original
train-original:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	WANDB_CACHE_DIR=$(WANDB_CACHE_DIR) \
	WANDB_DATA_DIR=$(WANDB_DATA_DIR) \
	python main.py \
		--config models/asf/config/asf-original.yaml \
		--train

.PHONY: train-asf
train-asf:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	WANDB_CACHE_DIR=$(WANDB_CACHE_DIR) \
	WANDB_DATA_DIR=$(WANDB_DATA_DIR) \
	python main.py \
		--config models/asf/config/asf-original-augment.yaml \
		--train

.PHONY: train-asf-gru
train-asf-gru:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	WANDB_CACHE_DIR=$(WANDB_CACHE_DIR) \
	WANDB_DATA_DIR=$(WANDB_DATA_DIR) \
	python main.py \
		--config models/asf/config/asf-gru.yaml \
		--train

.PHONY: train-asf-gru-vgg
train-asf-gru-vgg:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	WANDB_CACHE_DIR=$(WANDB_CACHE_DIR) \
	WANDB_DATA_DIR=$(WANDB_DATA_DIR) \
	python main.py \
		--config models/asf/config/asf-gru-vgg.yaml \
		--train

.PHONY: train-asf-gru-state-vgg
train-asf-gru-state-vgg:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	WANDB_CACHE_DIR=$(WANDB_CACHE_DIR) \
	WANDB_DATA_DIR=$(WANDB_DATA_DIR) \
	python main.py \
		--config models/asf/config/asf-gru-state-vgg.yaml \
		--train

.PHONY: train-asf-gru-aug-state-vgg
train-asf-gru-aug-state-vgg:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	WANDB_CACHE_DIR=$(WANDB_CACHE_DIR) \
	WANDB_DATA_DIR=$(WANDB_DATA_DIR) \
	python main.py \
		--config models/asf/config/asf-gru-augment-state-vgg.yaml \
		--train

.PHONY: train-asf-gru-aug
train-asf-gru-aug:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	WANDB_CACHE_DIR=$(WANDB_CACHE_DIR) \
	WANDB_DATA_DIR=$(WANDB_DATA_DIR) \
	python main.py \
		--config models/asf/config/asf-gru-augment.yaml \
		--train

.PHONY: train-asf-gru-aug-vgg
train-asf-gru-aug-vgg:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	WANDB_CACHE_DIR=$(WANDB_CACHE_DIR) \
	WANDB_DATA_DIR=$(WANDB_DATA_DIR) \
	python main.py \
		--config models/asf/config/asf-gru-augment-vgg.yaml \
		--train

.PHONY: train-asf-state
train-asf-state:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	WANDB_CACHE_DIR=$(WANDB_CACHE_DIR) \
	WANDB_DATA_DIR=$(WANDB_DATA_DIR) \
	python main.py \
		--config models/asf/config/asf-state.yaml \
		--train

.PHONY: train-asf-aug-state
train-asf-aug-state:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	WANDB_CACHE_DIR=$(WANDB_CACHE_DIR) \
	WANDB_DATA_DIR=$(WANDB_DATA_DIR) \
	python main.py \
		--config models/asf/config/asf-augment-state.yaml \
		--train

.PHONY: train-asf-aug-vgg
train-asf-aug-vgg:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	WANDB_CACHE_DIR=$(WANDB_CACHE_DIR) \
	WANDB_DATA_DIR=$(WANDB_DATA_DIR) \
	python main.py \
		--config models/asf/config/asf-augment-vgg.yaml \
		--train

.PHONY: train-asf-gru-state
train-asf-gru-state:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	WANDB_CACHE_DIR=$(WANDB_CACHE_DIR) \
	WANDB_DATA_DIR=$(WANDB_DATA_DIR) \
	python main.py \
		--config models/asf/config/asf-gru-state.yaml \
		--train

.PHONY: train-asf-gru-aug-state
train-asf-gru-aug-state:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	WANDB_CACHE_DIR=$(WANDB_CACHE_DIR) \
	WANDB_DATA_DIR=$(WANDB_DATA_DIR) \
	python main.py \
		--config models/asf/config/asf-gru-augment-state.yaml \
		--train


.PHONY: test
test:
	@$(CONDA_ACTIVATE) $(VENV_DIR)
	WANDB_CACHE_DIR=$(WANDB_CACHE_DIR) \
	WANDB_DATA_DIR=$(WANDB_DATA_DIR) \
	python main.py \
		--config models/asf/config/asf-augment.yaml \
		--test


.PHONY: job-train-asf
job-train-asf:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-train-asf.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time $(DURATION) \
	    --mem 16G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name asf-augment \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make train-asf"

.PHONY: job-train-asf-gru
job-train-asf-gru:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-train-asf-gru.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time $(DURATION) \
	    --mem 16G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name asf-gru \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make train-asf-gru"


.PHONY: job-train-asf-gru-aug
job-train-asf-gru-aug:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-train-asf-gru-aug.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time $(DURATION) \
	    --mem 64G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name asf-gru-aug \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make train-asf-gru-aug"


.PHONY: job-train-asf-state
job-train-asf-state:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	JOB_NAME=asf-state \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-$${JOB_NAME}.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time $(DURATION) \
	    --mem 16G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name $${JOB_NAME} \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make train-$${JOB_NAME}"

.PHONY: job-train-asf-gru-vgg
job-train-asf-gru-vgg:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	JOB_NAME=asf-gru-vgg \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-$${JOB_NAME}.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time $(DURATION) \
	    --mem 32G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name $${JOB_NAME} \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make train-$${JOB_NAME}"

.PHONY: job-train-asf-gru-aug-vgg
job-train-asf-gru-aug-vgg:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	JOB_NAME=asf-gru-aug-vgg \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-$${JOB_NAME}.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time $(DURATION) \
	    --mem 64G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name $${JOB_NAME} \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make train-$${JOB_NAME}"

.PHONY: job-train-original
job-train-original:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	JOB_NAME=original \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-$${JOB_NAME}.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time $(DURATION) \
	    --mem 16G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name $${JOB_NAME} \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make train-$${JOB_NAME}"

.PHONY: job-train-asf-aug-state
job-train-asf-aug-state:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	JOB_NAME=asf-aug-state \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-$${JOB_NAME}.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time $(DURATION) \
	    --mem 16G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name $${JOB_NAME} \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make train-$${JOB_NAME}"

.PHONY: job-train-asf-aug-vgg
job-train-asf-aug-vgg:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	JOB_NAME=asf-aug-vgg \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-$${JOB_NAME}.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time $(DURATION) \
	    --mem 16G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name $${JOB_NAME} \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make train-$${JOB_NAME}"

.PHONY: job-train-asf-gru-state
job-train-asf-gru-state:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	JOB_NAME=asf-gru-state \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-$${JOB_NAME}.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time $(DURATION) \
	    --mem 64G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name $${JOB_NAME} \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make train-$${JOB_NAME}"

.PHONY: job-train-asf-gru-aug-state
job-train-asf-gru-aug-state:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	JOB_NAME=asf-gru-aug-state \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-$${JOB_NAME}.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time $(DURATION) \
	    --mem 64G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name $${JOB_NAME} \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make train-$${JOB_NAME}"

.PHONY: job-train-asf-gru-aug-state-vgg
job-train-asf-gru-aug-state-vgg:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	JOB_NAME=asf-gru-aug-state-vgg \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-$${JOB_NAME}.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time $(DURATION) \
	    --mem 64G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name $${JOB_NAME} \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make train-$${JOB_NAME}"

.PHONY: job-train-asf-gru-state-vgg
job-train-asf-gru-state-vgg:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d-%T"); \
	JOB_NAME=asf-gru-state-vgg \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-$${JOB_NAME}.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time $(DURATION) \
	    --mem 64G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name $${JOB_NAME} \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make train-$${JOB_NAME}"



.PHONY: job-test
job-test:
	@mkdir -p $(LOGS_DIR)
	@DATE=$$(date +"%Y_%m_%d_%T"); \
	JOB_NAME=test-asf-gru; \
	LOG_FILE="$(REPO_DIR)/$(LOGS_DIR)/$${DATE}-$${JOB_NAME}.log"; \
	sbatch -N 1 \
	    --ntasks 1 \
	    --cpus-per-task 8 \
		--gres=gpu:1 \
	    --time 2:00:00 \
	    --mem 16G \
	    --error $${LOG_FILE} \
	    --output $${LOG_FILE} \
	    --job-name $${JOB_NAME} \
	    --open-mode append \
	    --mail-type "BEGIN,END" \
		--mail-user $(MAIL_ADDRESS) \
	    --wrap "cd $(REPO_DIR) && make test"
