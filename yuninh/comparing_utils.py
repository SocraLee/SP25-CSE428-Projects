from patient_utils import LabeledPatient
import constants as _
from file_utils import CSVHandler

class AccuracyCalculator:
    def __init__(self, labeled_patients):
        self.labeled_patients = labeled_patients
        self.wrong_direction = []
        self.results = {}
        self.wrong_data = []
        self.location_results = {_.Right: [], _.Wrong :[]}
        self.side_results = {_.Right: [], _.Wrong :[]}
        self.clock_results = {_.Right: [], _.Wrong :[]}
        self.adjusted_results = {_.Right: [], _.Wrong :[]}
        self.csv_handler = CSVHandler()

    def get_accuracy_without_weight(self, file_path):
        self.get_accuracy_with_weight(file_path, 1, 1, 1, 1)

    def get_accuracy_with_weight(self, file_path, location_weight, side_weight, clock_weight, adjusted_weight):
        total_weight = (location_weight + side_weight + clock_weight + adjusted_weight) * 1.0
        o_, output_labeled_patients = self.csv_handler.read_from_csv(file_path)
        for opid in output_labeled_patients:
            output_labeled_patient = output_labeled_patients[opid]
            curr_acc = 0
            if opid not in self.labeled_patients:
                raise KeyError("PatientId is invalid.")
            if output_labeled_patient == 0:
                self.results[opid] = curr_acc
                self.wrong_data.append(opid)
                self.location_results[_.Wrong].append(opid)
                self.side_results[_.Wrong].append(opid)
                self.clock_results[_.Wrong].append(opid)
                self.adjusted_results[_.Wrong].append(opid)
                continue
            l_p = self.labeled_patients[opid]
            if output_labeled_patient.direction != l_p.direction:
                self.results[opid] = curr_acc
                self.wrong_direction.append(opid)
                continue
            if output_labeled_patient.location == l_p.location:
                curr_acc += 100.0 * location_weight / total_weight
                self.location_results[_.Right].append(opid)
            else:
                self.location_results[_.Wrong].append(opid)
            if output_labeled_patient.side == l_p.side:
                curr_acc += 100.0 * side_weight / total_weight
                self.side_results[_.Right].append(opid)
            else:
                self.side_results[_.Wrong].append(opid)
            if output_labeled_patient.clock == l_p.clock:
                curr_acc += 100.0 * clock_weight / total_weight
                self.clock_results[_.Right].append(opid)
            else:
                self.clock_results[_.Wrong].append(opid)
            if output_labeled_patient.adjusted == l_p.adjusted:
                curr_acc += 100.0 * adjusted_weight / total_weight
                self.adjusted_results[_.Right].append(opid)
            else:
                self.adjusted_results[_.Wrong].append(opid)
            self.results[opid] = curr_acc

    def save_accuracy(self, file_path, file_name):
        contents = [["Accuracies Name", "Value"]]
        for pid in self.results:
            value = self.results[pid]
            content = [pid, value]
            contents.append(content)
        loc = ["Location Average Accuracie", self.get_values(self.location_results)]
        sid = ["Side Average Accuracie", self.get_values(self.side_results)]
        clo = ["Clock Average Accuracie", self.get_values(self.clock_results)]
        adj = ["Adjusted Average Accuracie", self.get_values(self.adjusted_results)]
        contents = contents + [loc, sid, clo, adj]
        self.csv_handler.write_into_csv(file_path, file_name, contents)
    
    def get_values(self, results):
        return len(results[_.Right])/(len(results[_.Right]) + len(results[_.Wrong])) * 1.0