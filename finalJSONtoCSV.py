import json
import csv
import glob
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil import parser
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class DataExtractor:
    @staticmethod
    def get_value(data, *keys):
        """Safely retrieves a value from a nested dictionary or list. Returns an empty string if any key is missing."""
        logger.debug(f"Attempting to retrieve value from keys: {keys}")
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            elif isinstance(data, list) and isinstance(key, int) and 0 <= key < len(data):
                data = data[key]
            else:
                logger.debug(f"Key {key} not found or data type mismatch. Returning empty string.")
                return ""  # Return empty string if the path does not exist
        logger.debug(f"Retrieved value: {data}")
        return data

    @staticmethod
    def get_timestamp_diff(data, start_key, end_key):
        """Calculates the difference between two timestamps in seconds. Returns an empty string if any key is missing or if parsing fails."""
        start = DataExtractor.get_value(data, start_key)
        end = DataExtractor.get_value(data, end_key)
        if start and end:
            try:
                timestamp_diff = (parser.parse(end) - parser.parse(start)).total_seconds()
                logger.debug(f"Calculated timestamp difference: {timestamp_diff} seconds")
                return timestamp_diff
            except Exception as e:
                logger.error(f"Error parsing timestamps: {e}")
                return ""
        logger.debug("One or both timestamps missing, returning empty string.")
        return ""

    @staticmethod
    def get_map_coordinate(data):
        """Retrieves the estimated map coordinates. Returns a list of empty strings if data is missing."""
        logger.debug("Attempting to retrieve map coordinates.")
        mapping_data = DataExtractor.get_value(data, "Sessions", "Mapping")
        if not mapping_data or len(mapping_data) < 1:
            logger.debug("Mapping data missing or empty.")
            return [""] * 12  # Return a list of 12 empty strings if data is missing

        xy_data = mapping_data[0].get("EstimatedCoordinates", {})
        logger.debug(f"Retrieved map coordinates: {xy_data}")
        return [
            xy_data.get("Nest_X", ""), xy_data.get("Nest_Y", ""),
            xy_data.get("Cave_X", ""), xy_data.get("Cave_Y", ""),
            xy_data.get("Arch_X", ""), xy_data.get("Arch_Y", ""),
            xy_data.get("Tree_X", ""), xy_data.get("Tree_Y", ""),
            xy_data.get("Volcano_X", ""), xy_data.get("Volcano_Y", ""),
            xy_data.get("Waterfall_X", ""), xy_data.get("Waterfall_Y", "")
        ]

    @staticmethod
    def get_map_coordinate_xy(data):
        """Retrieves X and Y coordinates for specific landmarks. Returns a list of empty strings if data is missing."""
        xy_data = DataExtractor.get_map_coordinate(data)
        logger.debug(f"Map coordinate XY data: {xy_data}")
        if not xy_data:
            return [""] * 12  # Return a list of empty strings for each coordinate pair
        return xy_data
    @staticmethod
    def get_pointing_judgement_data(data):
        pointing_data = DataExtractor.get_value(data, "Sessions", "Egocentric", 0, "PointingTasks")
        
        if not pointing_data:
            # If there is no pointing data, return 0 as the overall average
            logger.debug("No pointing data found. Returning 0 for overall average.")
            return ""

        task_data = {}
        all_errors = []
        for task in pointing_data:
            task_num = task.get("TaskNumber", len(task_data))
            judgements = task.get("PointingJudgements", [])
            errors = [j.get("Absolute_Error", 0) for j in judgements if "Absolute_Error" in j]
            if errors:
                task_data[task_num] = {
                    "errors": errors,
                    "average_error": sum(errors) / len(errors)
                }
                all_errors.extend(errors)
        overall_average = sum(all_errors) / len(all_errors) if all_errors else 0
        logger.debug(f"Calculated overall average error: {overall_average}")
        
        return overall_average
class JSONProcessor:
    def __init__(self, total_pi_trials, total_pointing_judgements, total_pointing_tasks, total_pt_trials):
        self.total_pi_trials = total_pi_trials
        self.total_pointing_judgements = total_pointing_judgements
        self.total_pointing_tasks = total_pointing_tasks
        self.total_pt_trials = total_pt_trials

    def process_file(self, file_path):
        try:
            logger.info(f"Processing file: {file_path}")
            with open(file_path, 'r') as file:
                data = json.load(file)
            logger.debug(f"Loaded JSON data: {data}")
            
            output = self.extract_data(data)
            logger.info(f"Processed file: {file_path}, output length: {len(output)}")
            return output
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
    
    def extract_data(self, data):
        logger.debug("Starting data extraction.")
        extractor = DataExtractor()
        output = []

        # Basic data
        try:
            output.extend([
                extractor.get_value(data, "MetaData", "Player_Name"),
                extractor.get_value(data, "Training", "phase1", "totalTime"),
                extractor.get_value(data, "Training", "phase2", "totalTime"),
                extractor.get_value(data, "Training", "phase3", "totalTime"),
                extractor.get_value(data, "Training", "phase5", "Trials", 0, "Data", "totalTime"),
                extractor.get_value(data, "Training", "phase5", "Trials", 1, "Data", "totalTime"),
            ])
            logger.debug(f"Extracted basic data: {output}")
        except Exception as e:
            logger.error(f"Error extracting basic data: {e}")

        # Calculate total homing time and total training time
        try:
            homing_time_1 = extractor.get_value(data, "Training", "phase5", "Trials", 0, "Data", "totalTime")
            homing_time_2 = extractor.get_value(data, "Training", "phase5", "Trials", 1, "Data", "totalTime")
            total_homing_time = calculate_total_time(homing_time_1, homing_time_2)
            output.append(total_homing_time)
            logger.debug(f"Calculated total homing time: {total_homing_time}")
        except Exception as e:
            logger.error(f"Error calculating total homing time: {e}")
        
        try:
            rotation_time = extractor.get_value(data, "Training", "phase1", "totalTime")
            movement_time = extractor.get_value(data, "Training", "phase2", "totalTime")
            total_training_time = calculate_total_time(rotation_time, movement_time, total_homing_time)
            output.append(total_training_time)
            logger.debug(f"Calculated total training time: {total_training_time}")
        except Exception as e:
            logger.error(f"Error calculating total training time: {e}")

        # Path Integration data
        try:
            pi_data = extractor.get_value(data, "Sessions", "PathIntegration", 0, "Trials")
            logger.debug(f"Extracted Path Integration data: {pi_data}")
            pi_totals, pi_distances, pi_dist_ratios, pi_final_angles, pi_corrected_angles = [], [], [], [], []

            if self.total_pi_trials > 0:
                for i in range(self.total_pi_trials):
                    if i < len(pi_data):
                        trial_data = pi_data[i]["Data"]
                        trial_values = [
                            trial_data.get("totalTime", ""),
                            trial_data.get("PIDistance", ""),
                            trial_data.get("PIDistanceRatio", ""),
                            trial_data.get("FinalPIAngle", ""),
                            trial_data.get("PIAngle", ""),
                            trial_data.get("CorrectedPIAngle", "")
                        ]
                        output.extend(trial_values)
                        pi_totals.append(trial_values[0])
                        pi_distances.append(trial_values[1])
                        pi_dist_ratios.append(trial_values[2])
                        pi_final_angles.append(trial_values[3])
                        pi_corrected_angles.append(trial_values[5])
                        logger.debug(f"Extracted trial {i} values: {trial_values}")
                    else:
                        output.extend([""] * 6)
                        logger.debug(f"Trial {i} data missing, filling with empty strings.")
            else:
                output.extend([""] * (self.total_pi_trials * 6))
                logger.debug("No PI trials, filled with empty strings.")
        except Exception as e:
            logger.error(f"Error extracting Path Integration data: {e}")

        try:
            pointing_data = extractor.get_value(data, "Sessions", "Egocentric", 0, "PointingTasks")
            logger.debug(f"Extracted Pointing Judgements data: {pointing_data}")
            pointing_errors = [[] for _ in range(self.total_pointing_tasks)]

            if self.total_pointing_tasks > 0:
                for i in range(self.total_pointing_tasks):
                    for j in range(self.total_pointing_judgements):
                        if i < len(pointing_data) and j < len(pointing_data[i]["PointingJudgements"]):
                            error = pointing_data[i]["PointingJudgements"][j].get("Absolute_Error", "")
                            output.append(error)
                            if error != "":
                                pointing_errors[i].append(float(error))
                            logger.debug(f"Extracted error for task {i}, judgement {j}: {error}")
                        else:
                            output.append("")
                            logger.debug(f"Error data missing for task {i}, judgement {j}, appending empty string.")
            else:
                output.extend([""] * (self.total_pointing_tasks * self.total_pointing_judgements))
                logger.debug("No Pointing Tasks, filled with empty strings.")
        except Exception as e:
            logger.error(f"Error extracting Pointing Judgements data: {e}")

        # Add overall average for pointing judgements
       
        try:
            overall_average = extractor.get_pointing_judgement_data(data)
            output.append(overall_average)
            logger.debug(f"Calculated overall average for pointing judgements: {overall_average}")
        except Exception as e:
            logger.error(f"Error calculating overall average for pointing judgements: {e}")

        # Remaining data
        try:
            remaining_data = [
                extractor.get_value(data, "Sessions", "Mapping", 0, "TotalTime"),
                extractor.get_timestamp_diff(data["Sessions"]["Mapping"][0], "StartTimeStamp", "EndTimeStamp") if extractor.get_value(data, "Sessions", "Mapping") else "",
                extractor.get_value(data, "Sessions", "Mapping", 0, "BidimensionalRegression", "Euclidean", "R2"),
                extractor.get_value(data, "Sessions", "Memory", 0, "TotalTime"),
                extractor.get_timestamp_diff(data["Sessions"]["Memory"][0], "StartTimeStamp", "EndTimeStamp") if extractor.get_value(data, "Sessions", "Memory") else "",
                extractor.get_value(data, "Sessions", "Memory", 0, "PercentCorrect"),
                extractor.get_value(data, "Sessions", "PerspectiveTaking", 0, "TotalIdleTime"),
                extractor.get_value(data, "Sessions", "PerspectiveTaking", 0, "TotalTime"),
                extractor.get_value(data, "Sessions", "PerspectiveTaking", 0, "AverageErrorMeasure"),
                extractor.get_value(data, "MetaData", "Start_Timestamp"),
                extractor.get_value(data, "MetaData", "End_Timestamp"),
                extractor.get_timestamp_diff(data["MetaData"], "Start_Timestamp", "End_Timestamp"),
            ]
            output.extend(remaining_data)
            logger.debug(f"Extracted remaining data: {remaining_data}")
        except Exception as e:
            logger.error(f"Error extracting remaining data: {e}")

        # Add map coordinate data
        try:
            map_data = extractor.get_map_coordinate_xy(data)
            output.extend(map_data)
            logger.debug(f"Extracted map coordinate data: {map_data}")
        except Exception as e:
            logger.error(f"Error extracting map coordinate data: {e}")

        # Perspective Taking data
        try:
            pt_data = extractor.get_value(data, "Sessions", "PerspectiveTaking", 0, "Trials")
            perspective_errors = []

            if self.total_pt_trials > 0:
                for i in range(self.total_pt_trials):
                    if i < len(pt_data):
                        trial_data = pt_data[i]
                        trial_values = [
                            trial_data.get("TotalTime", ""),
                            trial_data.get("TotalIdleTime", ""),
                            trial_data.get("FinalAngle", ""),
                            trial_data.get("CorrectAngle", ""),
                            trial_data.get("DifferenceAngle", ""),
                            trial_data.get("ErrorMeasure", "")
                        ]
                        output.extend(trial_values)
                        if trial_values[5] != "":
                            perspective_errors.append(float(trial_values[5]))
                        logger.debug(f"Extracted perspective trial {i} values: {trial_values}")
                    else:
                        output.extend([""] * 6)
                        logger.debug(f"Perspective trial {i} data missing, filling with empty strings.")
            else:
                output.extend([""] * (self.total_pt_trials * 6))
                logger.debug("No PT trials, filled with empty strings.")
        except Exception as e:
            logger.error(f"Error extracting Perspective Taking data: {e}")

        # Calculate and append averages for Path Integration data
        try:
            pi_averages = [
                sum(float(t) for t in pi_totals if t) / len(pi_totals) if pi_totals else "",
                sum(float(d) for d in pi_distances if d) / len(pi_distances) if pi_distances else "",
                sum(float(r) for r in pi_dist_ratios if r) / len(pi_dist_ratios) if pi_dist_ratios else "",
                sum(float(a) for a in pi_final_angles if a) / len(pi_final_angles) if pi_final_angles else "",
                sum(float(a) for a in pi_corrected_angles if a) / len(pi_corrected_angles) if pi_corrected_angles else ""
            ]
            output.extend(pi_averages)
            logger.debug(f"Calculated PI averages: {pi_averages}")
        except Exception as e:
            logger.error(f"Error calculating Path Integration averages: {e}")

        # Calculate and append averages for Pointing Judgements
        try:
            for errors in pointing_errors:
                avg_error = sum(errors) / len(errors) if errors else ""
                output.append(avg_error)
                logger.debug(f"Calculated average pointing error: {avg_error}")
        except Exception as e:
            logger.error(f"Error calculating Pointing Judgements averages: {e}")

        # Calculate and append average for Perspective Taking
        try:
            avg_perspective_error = sum(perspective_errors) / len(perspective_errors) if perspective_errors else ""
            output.append(avg_perspective_error)
            logger.debug(f"Calculated average perspective error: {avg_perspective_error}")
        except Exception as e:
            logger.error(f"Error calculating Perspective Taking average error: {e}")

        logger.info("Data extraction completed.")
        return output
    
    def calculate_pointing_judgement_total_time(self, data):
        pointing_tasks = DataExtractor.get_value(data, "Sessions", "Egocentric", 0, "PointingTasks")
        if not pointing_tasks:
            return ""
        
        first_timestamp = DataExtractor.get_value(pointing_tasks[0], "PointingJudgements", 0, "rawData", "Rotations", 0, "timeStamp")
        last_task = pointing_tasks[-1]
        last_judgement = last_task["PointingJudgements"][-1]
        last_timestamp = DataExtractor.get_value(last_judgement, "rawData", "Rotations", -1, "timeStamp")
        
        if first_timestamp and last_timestamp:
            return (parser.parse(last_timestamp) - parser.parse(first_timestamp)).total_seconds()
        return ""

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def get_column_headers(total_pi_trials, total_pointing_judgements, total_pointing_tasks, total_pt_trials):
    headers = [
        "Player_ID", "RotationTime", "MovementTime", "CircuitTime",
        "HomingTime_1", "HomingTime_2", "TotalHomingTime", "TotalTrainingTime"
    ]
    logger.debug(f"Initial headers: {headers}")

    # Add headers for PI trials if any
    if total_pi_trials > 0:
        logger.debug(f"Adding PI trial headers for {total_pi_trials} trials.")
        for i in range(total_pi_trials):
            headers.extend([
                f"PI_TotalTime_{i}", f"PI_Distance_{i}", f"PI_DistRatio_{i}",
                f"PI_FinalAngle_{i}", f"PI_Angle_{i}", f"PI_Corrected_PI_Angle_{i}"
            ])

    # Add headers for Pointing Tasks if any
    if total_pointing_tasks > 0:
        logger.debug(f"Adding Pointing Task headers for {total_pointing_tasks} tasks and {total_pointing_judgements} judgements.")
        for i in range(total_pointing_tasks):
            for j in range(total_pointing_judgements):
                headers.append(f"PointingJudgement_AbsoluteError_{i}_Trial_{j}")
            
    headers.append("Average_PointingJudgementError_all")
    headers.extend([
        "MapTotalTime", "CalculatedMapTotalTimeSeconds", "MapRSq",
        "MemoryTotalTime", "CalculatedMemoryTotalTimeSeconds", "MemoryPercentCorrect",
        "Overall_PerpectiveIdleTime", "Overall_PerspectiveTotalTime", "Overall_PerspectiveErrorMeasure",
        "SPACEStartTime", "SPACEEndTime", "SPACETotalTime"
    ])
    logger.debug(f"Headers after adding Pointing Tasks and other data: {headers}")

    # Add headers for map coordinates
    landmarks = ["Nest", "Cave", "Arch", "Tree", "Volcano", "Waterfall"]
    for landmark in landmarks:
        headers.extend([f"{landmark}_X", f"{landmark}_Y"])
    logger.debug(f"Headers after adding map coordinates: {headers}")

    # Add headers for Perspective Taking trials if any
    if total_pt_trials > 0:
        logger.debug(f"Adding Perspective Taking headers for {total_pt_trials} trials.")
        for i in range(total_pt_trials):
            headers.extend([
                f"PerspectiveTotalTime_{i}", f"PerpectiveIdleTime_{i}",
                f"PerpectiveFinalAngle_{i}", f"PerpectiveCorrectAngle_{i}",
                f"PerpectiveDifferenceAngle_{i}", f"PerspectiveErrorMeasure_{i}"
            ])

    # Add headers for average columns
    headers.extend([
        "Avg_PI_TotalTime", "Avg_PI_Distance", "Avg_PI_DistRatio",
        "Avg_PI_FinalAngle", "Avg_PI_Corrected_PI_Angle"
    ])
    logger.debug(f"Headers after adding average PI columns: {headers}")

    # Add headers for average Pointing Judgement errors if any tasks exist
    if total_pointing_tasks > 0:
        logger.debug(f"Adding average Pointing Judgement headers for {total_pointing_tasks} tasks.")
        headers.extend([f"Avg_PointingJudgement_AbsoluteError_{i}" for i in range(total_pointing_tasks)])

    headers.append("Avg_PerspectiveErrorMeasure")
    logger.debug(f"Final headers: {headers}")

    return headers

def calculate_pi_averages(df, select_columns, selected_trials=None):
    def get_pi_trial_indices(input_list):
        pi_trials = []
        for item in input_list:
            if 'PI (for each trial).PI_trial_' in item:
                trial_index = item.split('PI_trial_')[-1]
                pi_trials.append(int(trial_index))  # Convert to integer for proper sorting
        return sorted(pi_trials)
    
    pi_metrics = ["TotalTime", "Distance", "DistRatio", "FinalAngle", "Corrected_PI_Angle"]
    averages = {}
    selected_trials = get_pi_trial_indices(select_columns)
    logger.info(f"Selected_trials_PI: {selected_trials}")
                
    for metric in pi_metrics:
        if selected_trials:
            columns = [f"PI_{metric}_{i}" for i in selected_trials]
        else:
            columns = [col for col in df.columns if col.startswith(f"PI_{metric}_")]
        
        if columns:
            averages[f"Avg_PI_{metric}"] = df[columns].apply(lambda x: pd.to_numeric(x, errors='coerce')).mean(axis=1)
    
    return averages

import pandas as pd
import numpy as np
import logging

# Assuming logger is already configured
logger = logging.getLogger()

def calculate_pointing_averages(df, select_columns, total_num_pointing_trials):
    def get_pointing_trial_indices(input_list):
        pointing_trials = []
        for item in input_list:
            if 'Pointing error.Pointing_trial_' in item and 'PointingJudgement_AbsoluteError_' not in item:
                # Extract trial index
                trial_part = item.split('Pointing error.Pointing_trial_')[-1]
                if '.' not in trial_part:
                    pointing_trials.append(int(trial_part))  # Convert to integer for proper sorting
        return sorted(pointing_trials)

    selected_pointing_trials = get_pointing_trial_indices(select_columns)
    logger.info(f"Selected_trials_Pointing: {selected_pointing_trials}")

    def get_unselected_pointing_trials(total_num_pointing_trials, selected_columns):
        selected_pointing_trials = get_pointing_trial_indices(selected_columns)
        all_pointing_trials = list(range(total_num_pointing_trials))
        unselected_pointing_trials = [trial for trial in all_pointing_trials if trial not in selected_pointing_trials]
        return unselected_pointing_trials

    unselected_pointing_trials = get_unselected_pointing_trials(total_num_pointing_trials, select_columns)
    logger.info(f"UnSelected_trials_Pointing: {unselected_pointing_trials}")

    # Calculate new averages for selected trials
    for trial in selected_pointing_trials:
        selected_errors = [col for col in select_columns if f'PointingJudgement_AbsoluteError_{trial}_Trial_' in col]
        
        if selected_errors:
            valid_columns = [col.split('.')[-1] for col in selected_errors if col.split('.')[-1] in df.columns]
            # Calculate the average for the selected errors for the trial
            df[f'Avg_PointingJudgement_AbsoluteError_{trial}'] = df[valid_columns].apply(lambda x: pd.to_numeric(x, errors='coerce')).mean(axis=1)

    valid_trial_averages = []
    # Calculate the overall average based on selected trials
    for trial in selected_pointing_trials:
        if f'Avg_PointingJudgement_AbsoluteError_{trial}' in df.columns:
            trial_avg_column = df[f'Avg_PointingJudgement_AbsoluteError_{trial}']
            if not trial_avg_column.isna().all():
                valid_trial_averages.append(trial_avg_column)

    if valid_trial_averages:
        # Calculate the mean across all valid trial averages for each row
        df['Average_PointingJudgementError_all'] = pd.concat(valid_trial_averages, axis=1).apply(lambda x: pd.to_numeric(x, errors='coerce')).mean(axis=1)
    else:
        df['Average_PointingJudgementError_all'] = np.nan

    return unselected_pointing_trials, df['Average_PointingJudgementError_all']

def calculate_pet_averages(df, select_columns, selected_trials=None):
    def get_perspective_error__indices(input_list):
            perspective_trials = []
            for item in input_list:
                if 'Perspective taking.Perspective_trial_' in item:
                    # Extract trial index and ensure it's not including further subcolumns
                    trial_part = item.split('Perspective taking.Perspective_trial_')[-1]
                    if '.' not in trial_part:
                        perspective_trials.append(int(trial_part))  # Convert to integer for proper sorting
            return sorted(perspective_trials)
    selected_trials = get_perspective_error__indices(select_columns)
    logger.info(f"Selected_trials_PET: {selected_trials}")  
    if selected_trials:
        columns = [f"PerspectiveErrorMeasure_{i}" for i in selected_trials]
    else:
        columns = [col for col in df.columns if col.startswith(f"PerspectiveErrorMeasure_")]
        
    if columns:
        df["Avg_PerspectiveErrorMeasure"] = df[columns].apply(lambda x: pd.to_numeric(x, errors='coerce')).mean(axis=1)
        
       
def JSONtoCSV(json_files, csv_filename, total_pi_trials, total_pointing_judgements, total_pointing_tasks, total_pt_trials):
    logger.info(f"Processing {len(json_files)} JSON files")
    
    # Initialize JSON processor
    processor = JSONProcessor(total_pi_trials, total_pointing_judgements, total_pointing_tasks, total_pt_trials)
    
    # Generate column headers
    headers = get_column_headers(total_pi_trials, total_pointing_judgements, total_pointing_tasks, total_pt_trials)
    logger.debug(f"Generated column headers: {headers}")

    data = []
    for file_path in json_files:
        if file_path is not None:
            logger.info(f"Processing file: {file_path}")
            processed_data = processor.process_file(file_path)
            if processed_data is not None:
                data.append(processed_data)
                logger.debug(f"Processed data for file {file_path}: {processed_data}")
            else:
                logger.warning(f"No data returned for file: {file_path}")
    
    if not data:
        logger.warning("No valid data processed from any files.")
    else:
        logger.debug(f"Final data collected from all files: {data}")

    # Create DataFrame with collected data and headers
    try:
        df = pd.DataFrame(data, columns=headers)
        logger.info(f"DataFrame created with shape: {df.shape}")
        logger.debug(f"DataFrame columns: {df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Error creating DataFrame: {e}")
        return None

    # Save DataFrame to CSV
    try:
        df.to_csv(csv_filename, index=False)
        logger.info(f"Data saved to CSV file: {csv_filename}")
    except Exception as e:
        logger.error(f"Error saving CSV file: {e}")

    return df

    
    return df

def get_column_groups(df, total_pi_trials, total_pointing_judgements, total_pointing_tasks, total_pt_trials,selected_pi_trials=None):
    def findEstimatedLandmarks(df):
        landmarks = ['Nest_X', 'Nest_Y', 'Cave_X', 'Cave_Y', 'Arch_X', 'Arch_Y', 
                     'Tree_X', 'Tree_Y', 'Volcano_X', 'Volcano_Y', 'Waterfall_X', 'Waterfall_Y']
        return [landmark for landmark in landmarks if landmark in df.columns]

    estimated_landmarks = findEstimatedLandmarks(df)
    column_groups = {
        
        "Player": ["Player_ID"],
        "Training": [
            "RotationTime", "MovementTime", "CircuitTime",
            "TotalHomingTime", 'TotalTrainingTime'
        ],
        "PI (for each trial)": {
            "PI Summaries": {
                "PI TotalTime":["Avg_PI_TotalTime"],
                "PI Distance": ["Avg_PI_Distance"],
                "PI DistanceRatio":["Avg_PI_DistRatio"],
                "PI FinalAngle": ["Avg_PI_FinalAngle"],
                "Corrected PI Angle": ["Avg_PI_Corrected_PI_Angle"]
            }
        },
        "Pointing error": {
            "Pointing_error_averages":{
                "Error (every trial)": [f"Avg_PointingJudgement_AbsoluteError_{i}" for i in range(total_pi_trials)
                ],
                "Pointing_Error_Average_all":
                    [
                        "Average_PointingJudgementError_all"
                    ]
            }
        },
        "Map": {
            "MapTotalTime": ["MapTotalTime"],
            "MapRSq": ["MapRSq"],
            "EstimatedCoordinates": estimated_landmarks if estimated_landmarks else ["No landmarks found"]
        },
        "Memory": [
            'MemoryTotalTime', 'MemoryPercentCorrect'
        ],
        "Perspective taking": {
             "Pespective summaries":{
                "Perspective_Taking_Time": [
                    "Overall_PerspectiveTotalTime"
                ],
                "Perspective_Error_Average": [
                    "Avg_PerspectiveErrorMeasure"
                ], 
             }
        },
        "Overall Measures": [
            'SPACEStartTime', 'SPACEEndTime', 'SPACETotalTime'
        ]
    }

    # Group PI trial columns
    for i in range(total_pi_trials):
        pi_cols = [
            f'PI_TotalTime_{i}', f'PI_Distance_{i}', f'PI_DistRatio_{i}',
            f'PI_FinalAngle_{i}', f'PI_Corrected_PI_Angle_{i}'
        ]
        if any(col in df.columns for col in pi_cols):
            if isinstance(column_groups["PI (for each trial)"], dict):
                column_groups["PI (for each trial)"][f'PI_trial_{i}'] = pi_cols

    # Group Pointing error columns
    for i in range(total_pointing_tasks):
        pointing_cols = [f'PointingJudgement_AbsoluteError_{i}_Trial_{j}' for j in range(total_pointing_judgements)]
        if any(col in df.columns for col in pointing_cols):
            if isinstance(column_groups["Pointing error"], dict):
                column_groups["Pointing error"][f'Pointing_trial_{i}'] = pointing_cols

    # Group Perspective taking columns
    for i in range(total_pt_trials):
        perspective_cols = [
             f"PerspectiveErrorMeasure_{i}"
        ]
        if any(col in df.columns for col in perspective_cols):
            if isinstance(column_groups["Perspective taking"], dict):
                column_groups["Perspective taking"][f'Perspective_trial_{i}'] = perspective_cols

    return column_groups

def get_summary_columns():
    column_groups = {
        "Player": ["Player_ID"],
        "Training": [
            'TotalTrainingTime'
        ],
        "PI (for each trial)": {
            "PI Summaries": [
                "Avg_PI_TotalTime",
                "Avg_PI_Distance",
                "Avg_PI_FinalAngle",
            ]
        },
        "Pointing error": {
            "Pointing_Error_Average_all":
                [
                    "Average_PointingJudgementError_all"
                ]
        },
        "Map": {
            "MapRSq": ["MapRSq"],
        },
        "Memory": [
                'MemoryPercentCorrect'
        ],
        "Perspective taking": {
            "Pespective summaries": [
                "Avg_PerspectiveErrorMeasure"
            ]
        },
        "Overall Measures": [
            'SPACEStartTime', 'SPACEEndTime', 'SPACETotalTime'
        ]
        }
    return column_groups

def clean_column_groups(group, df): 
            if isinstance(group, dict):
                cleaned = {}
                for k, v in group.items():
                    cleaned_v = clean_column_groups(v, df)
                    if cleaned_v:
                        cleaned[k] = cleaned_v
                return cleaned
            elif isinstance(group, list):
                return [item for item in group if item in df.columns and not df[item].isna().all() and not (df[item] == '').all()]
            elif isinstance(group, str):
                return group if group in df.columns and not df[group].isna().all() and not (df[group] == '').all() else None
            return group

def calculate_total_time(*homing_times):
        # Convert empty strings to NaN for easier handling
        homing_times = [np.nan if t == "" else t for t in homing_times]

        # Check if all are NaN
        if all(pd.isna(t) for t in homing_times):
            total_homing_time = np.nan
        else:
            # Perform the summation, ignoring NaNs
            total_homing_time = sum(float(t) for t in homing_times if not pd.isna(t))
        
        return total_homing_time   