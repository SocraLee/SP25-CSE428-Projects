from pathlib import Path
from model_factory import ModelFactory
from file_utils import CSVHandler
from patient_utils import Direction
from datetime import datetime

class LabelingProcess:
    def __init__(self, unlabled_patients, model_name, algo):
        self.unlabled_patients = unlabled_patients
        self.model_name = model_name
        self.algo = algo
        self.current_model = ModelFactory(self.model_name, self.algo).get_model()
        self.output_path = f"{Path.cwd()}/outputs/{model_name}/Algorithm-v{algo}"
        self.now = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        self.output_name = f"{model_name}-Algorithm-v{algo}-output-{self.now}"
        self.headers = ["PID", 'Direction', 'Annotation']
        self.csv_handler = CSVHandler()
    
    def get_label_results(self, count):
        current_model = ModelFactory(self.model_name, self.algo).get_model()
        if count > len(self.unlabled_patients):
            count = self.unlabled_patients
        self.csv_handler.write_single_line_into_csv(self.output_path, self.output_name, self.headers)
        for i in range(count):
            p_i = self.unlabled_patients[i]
            current_model.build_request(p_i)
            res = current_model.get_output_from_given_model()
            print(f"Compelete processing of patient {p_i.patientId}")
            content = [p_i.patientId, Direction[p_i.direction].value, res]
            self.csv_handler.write_single_line_into_csv(self.output_path, self.output_name, content)
        print(f"The results are written in to the file {Path.cwd()}\\outputs\\{self.model_name}\\Algorithm-v{self.algo}\\{self.model_name}-Algorithm-v{self.algo}-output-{self.now}.csv")
        return f"{self.output_path}/{self.output_name}"