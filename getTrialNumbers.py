import re
import json
import pandas as pd
import zipfile
# Load the JSON file
# Define the pattern for matching columns
PATTERN_PATH_INTEGRATION = r"Sessions_PathIntegration_\d+_Trials_(\d+)"
PATTERN_POINTING_TASK = r"Sessions_Egocentric_\d+_PointingTasks_(\d+)"
PATTERN_POINTING_JUDGEMENT = r"Sessions_Egocentric_\d+_PointingTasks_\d+_PointingJudgements_(\d+)"
PATTERN_PERSPECTIVE_TAKING = r"Sessions_PerspectiveTaking_\d+_Trials_(\d+)"
PATTERN_LANDMARK = r"EstimatedCoordinates_(\w+)"
# Function to flatten nested dictionaries, excluding RawData
def flatten_json(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                if a == "RawData":  # Skip RawData entries
                    continue
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def process_single_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    flat_data = flatten_json(data)
    return pd.DataFrame([flat_data])

def process_zip_file(file_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        data = process_single_json(zip_ref.namelist()[0])
    flat_data = flatten_json(data)
    return pd.DataFrame([flat_data])
# Flatten the JSON data, excluding RawData

def process_input(file_path):
    if file_path.endswith('.zip'):
        print('this is a zip file')
        return process_zip_file(file_path)
    elif file_path.endswith('.json'):
        print('this is a single json file')
        return process_single_json(file_path)
    else:
        raise ValueError("Input file must be either a .zip or .json file")
    
def findMaximumTrial(pattern,file_path):
    df = process_input(file_path)
    csv_header = df.columns

    # Initialize variables
    max_j = 0
    min_j = 0
    matching_columns = []

    # Find matching columns and extract j values
    for col in csv_header:
        match = re.match(pattern, col)
        if match:
            j = int(match.group(1))
            matching_columns.append(col)
            max_j = max(max_j, j)
            min_j = min(min_j, j)
    # Print the results
    if matching_columns:
        print(f"Maximum j value: {max_j}" + " of" + pattern)
        print(f"Minimum j value: {min_j}"+ " of" + pattern)
    return max_j

def findMaximumTrial_df(pattern,df):
    csv_header = df.columns

    # Initialize variables
    max_j = 0
    min_j = 0
    matching_columns = []

    # Find matching columns and extract j values
    for col in csv_header:
        match = re.match(pattern, col)
        if match:
            j = int(match.group(1))
            matching_columns.append(col)
            max_j = max(max_j, j)
            min_j = min(min_j, j)
    # Print the results
    if matching_columns:
        print(f"Maximum j value: {max_j}" + " of" + pattern)
        print(f"Minimum j value: {min_j}"+ " of" + pattern)
    return max_j

def findAllTrials(file_path):
    num_pi = findMaximumTrial(PATTERN_PATH_INTEGRATION,file_path)
    num_pj = findMaximumTrial(PATTERN_POINTING_JUDGEMENT,file_path)
    num_pot = findMaximumTrial(PATTERN_POINTING_TASK,file_path)
    num_pet = findMaximumTrial(PATTERN_PERSPECTIVE_TAKING,file_path)
    print("Debug!!!!!usage=========================" + " " +  str(num_pi) + " " +  str(num_pj) + " " +  str(num_pot) + " " +  str(num_pet))
    return num_pi+1, num_pj+1, num_pot+1, num_pet+1

def findAllTrials_df(df):
    num_pi = findMaximumTrial_df(PATTERN_PATH_INTEGRATION,df)
    num_pj = findMaximumTrial_df(PATTERN_POINTING_JUDGEMENT,df)
    num_pot = findMaximumTrial_df(PATTERN_POINTING_TASK,df)
    num_pet = findMaximumTrial_df(PATTERN_PERSPECTIVE_TAKING,df)
    print("Debug!!!!!usage=========================" + " " +  str(num_pi) + " " +  str(num_pj) + " " +  str(num_pot) + " " +  str(num_pet))
    return num_pi+1, num_pj+1, num_pot+1, num_pet+1

def findEstimatedLandmarks(path_file):
    df = process_single_json(path_file)
    csv_headers = df.columns
    result = [re.search(PATTERN_LANDMARK, string).group(1) for string in csv_headers if re.search(PATTERN_LANDMARK, string)]
    return result

def count_pointing_judgements(df):
    csv_header = df.columns
    judgements_per_task = {}

    for col in csv_header:
        match = re.match(PATTERN_POINTING_JUDGEMENT, col)
        if match:
            task_num, judgement_num = map(int, match.groups())
            if task_num not in judgements_per_task:
                judgements_per_task[task_num] = set()
            judgements_per_task[task_num].add(judgement_num)

    return {task: len(judgements) for task, judgements in judgements_per_task.items()}
    
    