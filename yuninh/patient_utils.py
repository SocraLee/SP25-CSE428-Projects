from PIL import Image
from pathlib import Path
from enum import Enum

class Direction(Enum):
    LEFT = "LE"
    RIGHT = "RT"

class Location(Enum):
    Upper = "UPPER"
    Lower = "LOWER"
    Middle = "MIDDLE"

class Side(Enum):
    Inner = "INNER"
    Outer = "OUTER"
    Middle = "MIDDLE"

class UnLabledPatient:
    def __init__(self, patientId, direction):
        self.patientId = patientId
        self.direction = Direction(direction).name
        self.root_path = f"{Path.cwd()}/breast_cancer_raw_dataset/P_{patientId}/all/{Direction(direction).name}"
        self.cc = f"{self.root_path}/CC_calcification.jpg"
        self.mlo = f"{self.root_path}/MLO_calcification.jpg"

class LabeledPatient:
    def __init__(self, patientId, direction, location, side, clock, adjusted):
        self.patientId = patientId
        self.direction = Direction(direction).name
        self.location = Location(location).name
        self.side = Side(side).name
        self.clock = float(clock)
        self.adjusted = int(adjusted)
        