from pathlib import Path
import os
import csv
from patient_utils import UnLabledPatient, LabeledPatient

class CSVHandler:
    def __init__(self):
        pass
    
    def read_from_csv(self, input_path):
        unlabled_patients = []
        labled_patients = {}
        with open(input_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row['PID']
                direction = row['Direction']
                annotation = row['Annotation']
                ul_p = UnLabledPatient(pid, direction)
                r = annotation.split()
                if (len(r) == 6):
                    l_p = LabeledPatient(pid, r[0], r[1], r[2], r[3], r[4])
                else:
                    l_p = 0
                unlabled_patients.append(ul_p)
                labled_patients[pid] = l_p
        return unlabled_patients, labled_patients


    def write_single_line_into_csv(self, output_path, output_name, content):
        os.makedirs(output_path, exist_ok=True)
        with open(f"{output_path}/{output_name}.csv", "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(content)

    def write_into_csv(self, output_path, output_name, contents):
        os.makedirs(output_path, exist_ok=True)
        with open(f"{output_path}/{output_name}.csv", "a", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(contents)