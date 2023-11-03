PYTHON=3.9
BASENAME=$(shell basename $(CURDIR))

env:
	conda create -n $(BASENAME) -y python=$(PYTHON)

setup:
	pip install uvicorn==0.23.2
	pip install segment_anything==1.0
	pip install torch==2.1.0
	pip install torchvision==0.16.0
	pip install openai==0.28.1
	pip install diffusers==0.21.4
