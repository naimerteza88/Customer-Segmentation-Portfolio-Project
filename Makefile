.PHONY: install data train test app docker

install:
	python -m pip install -r requirements.txt

data:
	python scripts/generate_data.py

train:
	PYTHONPATH=src python -m customer_segmentation.train

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

app:
	PYTHONPATH=src streamlit run app.py

docker:
	docker build -t customer-segmentation .

