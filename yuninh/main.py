from labeling_process import LabelingProcess
import constants as _
from pathlib import Path
from file_utils import CSVHandler
from comparing_utils import AccuracyCalculator

file = CSVHandler()
unlabled_patients, labeled_patients = file.read_from_csv(f"{Path.cwd()}/Breast Cancer Annotation - Dataset.csv")
l_p_0 = LabelingProcess(unlabled_patients, _.QwenMaxModelName, _.BasePrompt)
l_p_1 = LabelingProcess(unlabled_patients, _.QwenMaxModelName, _.OptimizedV1)
l_p_2 = LabelingProcess(unlabled_patients, _.QwenPlusModelName, _.BasePrompt)
l_p_3 = LabelingProcess(unlabled_patients, _.QwenPlusModelName, _.OptimizedV1)

result_name_and_path_0 = l_p_0.get_label_results(1)
ac0 = AccuracyCalculator(labeled_patients)
ac0.get_accuracy_without_weight(f"{l_p_0.output_path}/{l_p_0.output_name}.csv")
ac0.save_accuracy(l_p_0.output_path, f"{l_p_0.model_name}-accuracies-v{_.BasePrompt}-{l_p_0.now}")

result_name_and_path_1 = l_p_1.get_label_results(1)
ac1 = AccuracyCalculator(labeled_patients)
ac1.get_accuracy_without_weight(f"{l_p_1.output_path}/{l_p_1.output_name}.csv")
ac1.save_accuracy(l_p_1.output_path, f"{l_p_1.model_name}-accuracies-v{_.OptimizedV1}-{l_p_1.now}")

result_name_and_path_2 = l_p_2.get_label_results(1)
ac2 = AccuracyCalculator(labeled_patients)
ac2.get_accuracy_without_weight(f"{l_p_2.output_path}/{l_p_2.output_name}.csv")
ac2.save_accuracy(l_p_2.output_path, f"{l_p_2.model_name}-accuracies-v{_.BasePrompt}-{l_p_2.now}")

result_name_and_path_3 = l_p_3.get_label_results(1)
ac3 = AccuracyCalculator(labeled_patients)
ac3.get_accuracy_without_weight(f"{l_p_3.output_path}/{l_p_3.output_name}.csv")
ac3.save_accuracy(l_p_3.output_path, f"{l_p_3.model_name}-accuracies-v{_.OptimizedV1}-{l_p_3.now}")