1. Pre-setup
DashScope Python SDK equal or larger than 1.20.7

2. Input
To use the local image.
Please change the class UnlabeledPatient in the file patient_utils.py

3. To run the whole program, please run the command 
python .\main.py
or go run in main_workflow.ipynb chunck by chunck

4. Future development
	a. Introduce new algorithm
		1) Add system context in the directory system_context with name {model_name}_v{version_number}.txt
			i. 0 means base prompt.
		2) If the version is higher than 2, please add new constant of the version name in constants.py file
		3) Add user content in the directory user_context with name {model_name}_v{version_number}.txt
	b. Add new models
		1) Same steps first to create algorithm for this model
		2) Add new Model implementation .py file (e.g. qwen_model.py)
		3) Register new model in model_factory.py
		4) Use this model in labeling_process.py
	c. Add training
		1) change in qwen_model.py with build requests
		2) TBD