import constants
import dashscope
from dashscope import MultiModalConversation
from pathlib import Path
import os
from base_model import BaseModel
import re
from file_utils import CSVHandler

class QwenModel(BaseModel):
    def __init__(self, algo, model_name):
        super().__init__(algo)
        self.messages = []
        self.model_name = model_name

    def get_system_context(self):
        with open(f"{Path.cwd()}/system_context/qwen_v{self.algo}.txt", "r", encoding="utf-8") as f:
            system_contect = f.read()
        return system_contect
    
    def get_user_content(self):
        with open(f"{Path.cwd()}/user_content/qwen_v{self.algo}.txt", "r", encoding="utf-8") as f:
            system_contect = f.read()
        return system_contect
    
    def build_request(self, unlabeled_patient):
        messages = [
            {
                "role": "system",
                "content": [{"text": self.get_system_context()}]
            },
            {
                "role": "user",
                "content": [
                    {"image": unlabeled_patient.mlo},
                    {"image": unlabeled_patient.cc},
                    {"text": f"""This is patient with PID {unlabeled_patient.patientId} and the direction is {unlabeled_patient.direction}.
                     The first image is the cc and the second image is mlo. 
                     Please help me determine the output based on the following steps {self.get_user_content()}.
                     {constants.UserContentSuffix}"""}
                    ]
            }
        ]
        self.messages = messages
        return messages

    def get_output_from_given_model(self):
        response = dashscope.MultiModalConversation.call(
            api_key = "sk-f6d56915635a4bb4b67872dd1007ef8b",
            model = self.model_name, 
            messages = self.messages
        )
        res_content = response.output.choices[0].message.content[0]["text"]
        result = re.findall(r'Thus, the final result is:\s*(?:\*{2})?(.*?)(?:\*{2})?$', res_content)
        if len(result) > 0:
            line = result[0]
            while line.startswith(constants.ResultPrefix):
                line = line[len(constants.ResultPrefix):]
            return line
        