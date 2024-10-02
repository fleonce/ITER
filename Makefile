submission: convert.py evaluate.py train.py iter cfg scripts
	git archive --format=zip HEAD -o submission.zip

ade_configs:
	python3 scripts/datasets/load_ade.py

clean:
	rm -vf datasets/*/*.safetensors
	rm -vf datasets/*/*_filtered.json
	rm -vf datasets/*/*_entities.json
	rm -rvf build
	rm -vf submission.zip