from patient_utils import LabeledPatient

class BaseModel:
    def __init__(self, algo):
        self.algo = algo

    def get_output_from_given_model(self):
        return LabeledPatient("", "LE", "UPPER", "INNER", 1.0, 1)
    
    def get_system_context(self):
        return ""
    
    def get_user_content(self):
        return ""
    
    def build_request(self, r):
        return []
    
    def write_result_to_text_file(self, content):
        return 