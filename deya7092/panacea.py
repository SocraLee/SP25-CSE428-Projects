# Panacea Evaluation Notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import re

# === Embedded JSON Data ===

reference_trials = {
  "0": {
    "population": "Adults aged 18-65 with hypertension",
    "intervention": "New antihypertensive drug X",
    "outcome": "Reduction in systolic blood pressure",
    "eligibility": "No prior cardiovascular events, not pregnant",
    "phase": "Phase II"
  },
  "1": {
    "population": "Patients with Type 2 Diabetes",
    "intervention": "Dietary counseling plus drug Y",
    "outcome": "HbA1c decrease after 6 months",
    "eligibility": "Age 30-70, HbA1c > 7.0%, no renal failure",
    "phase": "Phase III"
  },
  "2": {
    "population": "Children with asthma aged 6-12",
    "intervention": "Inhaled corticosteroid Z",
    "outcome": "Reduced asthma exacerbations",
    "eligibility": "No other respiratory diseases",
    "phase": "Phase II"
  },
  "3": {
    "population": "Elderly patients > 65 with osteoarthritis",
    "intervention": "Physical therapy regimen",
    "outcome": "Improved joint mobility",
    "eligibility": "No recent joint surgery",
    "phase": "Phase IV"
  },
  "4": {
    "population": "Adults with chronic migraine",
    "intervention": "Monthly injection of drug A",
    "outcome": "Decrease in migraine days per month",
    "eligibility": "At least 4 migraine days per month",
    "phase": "Phase III"
  }
}

panacea_trial_outputs = {
  "0": {
    "population": "Adults 18-65 with high blood pressure",
    "intervention": "Drug X administration",
    "outcome": "Lower systolic BP",
    "eligibility": "Exclude prior heart issues, exclude pregnant women",
    "phase": "Phase II"
  },
  "1": {
    "population": "Type 2 diabetic patients",
    "intervention": "Drug Y with diet advice",
    "outcome": "Reduction in HbA1c",
    "eligibility": "Age between 30 and 70, no kidney failure",
    "phase": "Phase III"
  },
  "2": {
    "population": "Children 6 to 12 years with asthma",
    "intervention": "Inhaled steroid Z",
    "outcome": "Fewer asthma attacks",
    "eligibility": "No other lung diseases",
    "phase": "Phase II"
  },
  "3": {
    "population": "Older adults above 65 with osteoarthritis",
    "intervention": "Physical therapy exercises",
    "outcome": "Better joint function",
    "eligibility": "No recent surgeries on joints",
    "phase": "Phase IV"
  },
  "4": {
    "population": "Adults suffering chronic migraines",
    "intervention": "Monthly injection of drug A",
    "outcome": "Reduced migraine frequency",
    "eligibility": "Minimum 4 migraines monthly",
    "phase": "Phase III"
  }
}

patient_profiles = {
  "0": {
    "id": "0",
    "age": 55,
    "diagnosis": "hypertension",
    "medical_history": ["none"],
    "current_medication": []
  },
  "1": {
    "id": "1",
    "age": 50,
    "diagnosis": "type 2 diabetes",
    "medical_history": ["high cholesterol"],
    "current_medication": ["metformin"]
  },
  "2": {
    "id": "2",
    "age": 10,
    "diagnosis": "asthma",
    "medical_history": ["seasonal allergies"],
    "current_medication": ["albuterol inhaler"]
  },
  "3": {
    "id": "3",
    "age": 70,
    "diagnosis": "osteoarthritis",
    "medical_history": ["hip replacement"],
    "current_medication": ["ibuprofen"]
  },
  "4": {
    "id": "4",
    "age": 35,
    "diagnosis": "chronic migraine",
    "medical_history": ["depression"],
    "current_medication": ["topiramate"]
  }
}

panacea_patient_matches = {
  "0": {
    "patient_id": "0",
    "matched_trials": ["0"]
  },
  "1": {
    "patient_id": "1",
    "matched_trials": ["1"]
  },
  "2": {
    "patient_id": "2",
    "matched_trials": ["2"]
  },
  "3": {
    "patient_id": "3",
    "matched_trials": ["3"]
  },
  "4": {
    "patient_id": "4",
    "matched_trials": ["4"]
  }
}

chatgpt_reviews = {
    "trial_design_reviews": {
        "0": {
            "score": 4.5,
            "review": "The trial design aligns well with the reference. It clearly identifies adults with hypertension and includes a logical outcome (lower systolic BP). The eligibility criteria reasonably exclude confounding factors like prior heart issues and pregnancy. The phase is correct. Minor wording differences exist, but core details are retained."
        },
        "1": {
            "score": 3.8,
            "review": "The design captures key elements like age range and comorbidity (Type 2 Diabetes), but lacks clarity in eligibility phrasing and simplifies the outcome. Including 'diet advice' is acceptable but more detail would help. Phase is correctly retained."
        },
        "2": {
            "score": 4.2,
            "review": "This trial preserves the essential structure and medical focus. The population, intervention, and outcome are all appropriately translated. A bit more detail in eligibility would improve accuracy, but phase and treatment rationale are sound."
        },
        "3": {
            "score": 3.5,
            "review": "The trial targets the right population (elderly with osteoarthritis), and the intervention is generally accurate. However, 'better joint function' is vague compared to 'improved joint mobility'. Eligibility is correct but could use firmer clinical language."
        },
        "4": {
            "score": 4.7,
            "review": "Excellent match. It keeps all core clinical aspects from the reference trial, including a well-specified outcome and matching phase. Eligibility and intervention phrasing are very faithful to the original."
        }
    },
    "patient_match_reviews": {
        "0": {
            "score": 3.9,
            "review": "Patient 0 is correctly matched to a trial for hypertension in their age group. Medical history exclusions are respected. Slight vagueness in age specificity could be improved."
        },
        "1": {
            "score": 3.5,
            "review": "The match is acceptable given the diagnosis and age, but the trial eligibility includes exclusions for renal failure, which are not explicitly checked against the patient's profile. Minor risk of mismatch."
        },
        "2": {
            "score": 4.0,
            "review": "Good match. Asthmatic child within specified age range, no conflicting medical history. Exclusion criteria are respected."
        },
        "3": {
            "score": 3.2,
            "review": "Reasonable match by diagnosis and age, but the patient has a history of joint replacement, which might qualify as 'recent joint surgery' depending on timing. A stronger check on exclusions would help."
        },
        "4": {
            "score": 4.1,
            "review": "Strong match. Patient meets the migraine frequency criterion and is in the expected age group. No exclusion triggers are noted."
        }
    }
}

# === Trial Design Evaluation Function ===

def score_trial_design(ref, pred):
    """
    Scores trial design predictions (0-5 scale) considering:
    - presence & quality of population, intervention, outcome, eligibility
    - correct trial phase
    - explicit measurement/outcome detail
    - penalizes vague eligibility (missing age ranges)
    - penalizes intervention not aligned with population (condition)
    """
    score = 0
    keys = ['population', 'intervention', 'outcome', 'eligibility']

    # Check presence and overlap of content
    for k in keys:
        if k in pred and k in ref:
            if len(pred[k]) > 0 and len(ref[k]) > 0:
                pred_words = set(pred[k].lower().split())
                ref_words = set(ref[k].lower().split())
                if len(pred_words & ref_words) > 1:
                    score += 1

    # Check phase match
    if 'phase' in pred and 'phase' in ref:
        if pred['phase'].strip().lower() == ref['phase'].strip().lower():
            score += 1

    # Check outcome measurement keywords
    outcome_keywords = ['reduction', 'decrease', 'improve', 'lower', 'fewer', 'increase']
    outcome_text = pred.get('outcome', '').lower()
    if any(word in outcome_text for word in outcome_keywords):
        score += 1

    # Penalize missing age range if ref has it
    age_range_ref = re.findall(r'(\d+)-(\d+)', (ref.get('eligibility','') + ref.get('population','')))
    age_range_pred = re.findall(r'(\d+)-(\d+)', (pred.get('eligibility','') + pred.get('population','')))
    if age_range_ref and not age_range_pred:
        score -= 0.5

    # Penalize intervention not mentioning condition keywords
    diagnosis_terms = ref.get('population','').lower().split()
    intervention_terms = pred.get('intervention','').lower().split()
    if not any(term in intervention_terms for term in diagnosis_terms):
        score -= 0.5

    # Clamp score to 0-5 range
    score = max(0, min(5, score))
    return round(score, 2)

# === Score all trial designs ===

trial_scores = []
for key in reference_trials:
    ref = reference_trials[key]
    pred = panacea_trial_outputs.get(key, {})
    s = score_trial_design(ref, pred)
    trial_scores.append({'id': key, 'auto_score': s})

trial_df = pd.DataFrame(trial_scores)

# === Patient-Trial Matching Evaluation Function ===

def score_patient_matching(patient, matched_trial_ids, reference_trials):
    """
    Scores patient-trial matching (0-5 scale) based on:
    - condition and age match
    - coverage (at least one trial matched)
    - exclusion criteria respected (e.g. medical history)
    - penalizes too few matches or overly vague eligibility
    """
    if not matched_trial_ids:
        return 0

    relevant = 0
    exclusion_violated = False
    total_trials = len(matched_trial_ids)

    for tid in matched_trial_ids:
        trial = reference_trials.get(tid, {})
        pop = trial.get('population', '').lower()
        elig = trial.get('eligibility', '').lower()
        diagnosis = patient.get('diagnosis', '').lower()
        age = patient.get('age', 0)
        med_hist = [h.lower() for h in patient.get('medical_history', [])]
        current_meds = [m.lower() for m in patient.get('current_medication', [])]

        # Condition match
        condition_match = diagnosis in pop or any(d in pop for d in diagnosis.split())

        # Age check if age range present
        age_range = re.findall(r'(\d+)-(\d+)', pop)
        if age_range:
            low, high = map(int, age_range[0])
            age_ok = (low <= age <= high)
        else:
            age_ok = True  # no age info, assume okay

        # Check exclusion violations (simple heuristic)
        exclusions = ['renal failure', 'kidney failure', 'recent surgery', 'pregnant', 'cardiovascular events']
        for excl in exclusions:
            if excl in elig and any(excl.split()[0] in mh for mh in med_hist):
                exclusion_violated = True

        if condition_match and age_ok and not exclusion_violated:
            relevant += 1

    coverage = 1 if total_trials >= 1 else 0

    # Penalize vague eligibility (<10 chars)
    vague_eligibility = any(len(reference_trials[tid].get('eligibility', '')) < 10 for tid in matched_trial_ids)

    score = 0
    if relevant == total_trials and total_trials > 0:
        score += 2
    score += 2 * coverage
    if not exclusion_violated:
        score += 1
    if vague_eligibility:
        score -= 0.5

    score = max(0, min(5, score))
    return round(score, 2)

# === Score all patient-trial matches ===

patient_scores = []
for pid in patient_profiles:
    patient = patient_profiles[pid]
    matched = panacea_patient_matches.get(pid, {}).get('matched_trials', [])
    s = score_patient_matching(patient, matched, reference_trials)
    patient_scores.append({'id': pid, 'auto_score': s})

matching_df = pd.DataFrame(patient_scores)

# === Extract ChatGPT Scores into Simple Dicts for Comparison ===

chatgpt_scores = {
    'trial_design_scores': {
        k: v['score'] for k, v in chatgpt_reviews['trial_design_reviews'].items()
    },
    'patient_matching_scores': {
        k: v['score'] for k, v in chatgpt_reviews['patient_match_reviews'].items()
    }
}

# === Integrate ChatGPT Scores ===

trial_df['chatgpt_score'] = trial_df['id'].apply(
    lambda x: chatgpt_scores['trial_design_scores'].get(str(x), np.nan)
)

matching_df['chatgpt_score'] = matching_df['id'].apply(
    lambda x: chatgpt_scores['patient_matching_scores'].get(str(x), np.nan)
)

# === Compute BLEU and ROUGE as Baselines for Trial Design (population only) ===

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smooth_fn = SmoothingFunction().method1  # smoothing function to avoid zero BLEU

bleu_scores = []
rougeL_scores = []

for key in reference_trials:
    ref_text = reference_trials[key]['population'].lower().split()
    pred_text = panacea_trial_outputs[key]['population'].lower().split()
    bleu = sentence_bleu([ref_text], pred_text, smoothing_function=smooth_fn)
    bleu_scores.append(bleu)

    rouge_scores = scorer.score(reference_trials[key]['population'], panacea_trial_outputs[key]['population'])
    rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

trial_df['bleu'] = bleu_scores
trial_df['rougeL'] = rougeL_scores

# === Visualizations ===

def plot_comparisons(trial_df, matching_df):
    sns.set(style="whitegrid")

    # --- Trial Design: Proposed Task-Specific Metric Scores ---
    plt.figure(figsize=(10, 5))
    x = trial_df['id'].astype(int)
    width = 0.3

    plt.bar(x - width/2, trial_df['auto_score'], width=width, label='Auto Score', color='skyblue')
    plt.bar(x + width/2, trial_df['chatgpt_score'], width=width, label='ChatGPT Score', color='salmon')

    plt.xlabel('Trial ID')
    plt.ylabel('Proposed Task-Specific Metric Score (1–5)')
    plt.title('Trial Design: Proposed Task-Specific Metric Evaluation Scores')
    plt.xticks(x)
    plt.ylim(0, 5.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Trial Design: BLEU & ROUGE ---
    plt.figure(figsize=(10, 5))
    width = 0.3

    plt.bar(x - width/2, trial_df['bleu'], width=width, label='BLEU', color='lightgreen')
    plt.bar(x + width/2, trial_df['rougeL'], width=width, label='ROUGE-L', color='orange')

    plt.xlabel('Trial ID')
    plt.ylabel('Score (0–1)')
    plt.title('Trial Design: Text Similarity Metrics')
    plt.xticks(x)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- Patient-Trial Matching: Proposed Task-Specific Metric Scores ---
    plt.figure(figsize=(8, 5))
    x = matching_df['id'].astype(int)
    width = 0.3

    plt.bar(x - width/2, matching_df['auto_score'], width=width, label='Auto Score', color='skyblue')
    plt.bar(x + width/2, matching_df['chatgpt_score'], width=width, label='ChatGPT Score', color='salmon')

    plt.xlabel('Patient ID')
    plt.ylabel('Proposed Task-Specific Metric Score (1–5)')
    plt.title('Patient-Trial Matching: Proposed Task-Specific Metric Evaluation Scores')
    plt.xticks(x)
    plt.ylim(0, 5.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Print Results ===

print("\nTrial Design Evaluation:")
print(trial_df)

print("\nPatient-Trial Matching Evaluation:")
print(matching_df)

print("\nAverage Scores:")
print(f"Trial Design - Auto: {trial_df['auto_score'].mean():.2f}, ChatGPT: {trial_df['chatgpt_score'].mean():.2f}")
print(f"Trial Design - BLEU: {trial_df['bleu'].mean():.3f}, ROUGE-L: {trial_df['rougeL'].mean():.3f}")
print(f"Patient Matching - Auto: {matching_df['auto_score'].mean():.2f}, ChatGPT: {matching_df['chatgpt_score'].mean():.2f}")
plot_comparisons(trial_df, matching_df)
