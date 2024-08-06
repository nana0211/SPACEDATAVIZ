import json
import csv
import glob
import os
import logging
import pandas as pd
from datetime import datetime
from dateutil import parser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataExtractor:
    @staticmethod
    def get_value(data, *keys):
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            elif isinstance(data, list) and isinstance(key, int) and 0 <= key < len(data):
                data = data[key]
            else:
                return ""
        return data

    @staticmethod
    def get_timestamp_diff(data, start_key, end_key):
        start = DataExtractor.get_value(data, start_key)
        end = DataExtractor.get_value(data, end_key)
        if start and end:
            return (parser.parse(end) - parser.parse(start)).total_seconds()
        return ""

    @staticmethod
    def get_map_coordinate(data):
        mapping_data = DataExtractor.get_value(data, "Sessions", "Mapping")
        if not mapping_data or len(mapping_data) < 1:
            return ""
        return mapping_data[0].get("EstimatedCoordinates", "")

    @staticmethod
    def get_map_coordinate_xy(data):
        xy_data = DataExtractor.get_map_coordinate(data)
        if not xy_data:
            return []
        
        map_data = []
        landmarks = ["Nest", "Cave", "Arch", "Tree", "Volcano", "Waterfall"]
        
        for landmark in landmarks:
            if landmark in xy_data:
                map_data.extend([xy_data[landmark].get("X", ""), xy_data[landmark].get("Y", "")])
        
        return map_data

class JSONProcessor:
    def __init__(self, total_pi_trials, total_pointing_judgements, total_pointing_tasks, total_pt_trials):
        self.total_pi_trials = total_pi_trials
        self.total_pointing_judgements = total_pointing_judgements
        self.total_pointing_tasks = total_pointing_tasks
        self.total_pt_trials = total_pt_trials

    def process_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            output = self.extract_data(data)
            logger.info(f"Processed file: {file_path}, output length: {len(output)}")
            return output
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None

    def extract_data(self, data):
        extractor = DataExtractor()
        output = []

        # Basic data
        output.extend([
            extractor.get_value(data, "MetaData", "Player_Name"),
            extractor.get_value(data, "Training", "phase1", "totalTime"),
            extractor.get_value(data, "Training", "phase2", "totalTime"),
            extractor.get_value(data, "Training", "phase3", "totalTime"),
            extractor.get_value(data, "Training", "phase5", "Trials", 0, "Data", "totalTime"),
            extractor.get_value(data, "Training", "phase5", "Trials", 1, "Data", "totalTime"),
        ])
        
        # Calculate total homing time and total training time
        homing_time_1 = extractor.get_value(data, "Training", "phase5", "Trials", 0, "Data", "totalTime")
        homing_time_2 = extractor.get_value(data, "Training", "phase5", "Trials", 1, "Data", "totalTime")
        total_homing_time = sum(float(t) for t in [homing_time_1, homing_time_2] if t)
        output.append(total_homing_time)
        
        rotation_time = extractor.get_value(data, "Training", "phase1", "totalTime")
        movement_time = extractor.get_value(data, "Training", "phase2", "totalTime")
        total_training_time = sum(float(t) for t in [rotation_time, movement_time, total_homing_time] if t)
        output.append(total_training_time)

        # Path Integration data
        pi_data = extractor.get_value(data, "Sessions", "PathIntegration", 0, "Trials")
        for i in range(self.total_pi_trials):
            if i < len(pi_data):
                trial_data = pi_data[i]["Data"]
                output.extend([
                    trial_data.get("totalTime", ""),
                    trial_data.get("PIDistance", ""),
                    trial_data.get("PIDistanceRatio", ""),
                    trial_data.get("FinalPIAngle", ""),
                    trial_data.get("PIAngle", ""),
                    trial_data.get("CorrectedPIAngle", "")
                ])
            else:
                output.extend([""] * 6)

        # Pointing Judgements data
        pointing_data = extractor.get_value(data, "Sessions", "Egocentric", 0, "PointingTasks")
        for i in range(self.total_pointing_tasks):
            for j in range(self.total_pointing_judgements):
                if i < len(pointing_data) and j < len(pointing_data[i]["PointingJudgements"]):
                    output.append(pointing_data[i]["PointingJudgements"][j].get("Absolute_Error", ""))
                else:
                    output.append("")

        # Remaining data
        output.extend([
            self.calculate_pointing_judgement_total_time(data),
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
        ])

        # Add map coordinate data
        output.extend(extractor.get_map_coordinate_xy(data))

        # Perspective Taking data
        pt_data = extractor.get_value(data, "Sessions", "PerspectiveTaking", 0, "Trials")
        for i in range(self.total_pt_trials):
            if i < len(pt_data):
                trial_data = pt_data[i]
                output.extend([
                    trial_data.get("TotalTime", ""),
                    trial_data.get("TotalIdleTime", ""),
                    trial_data.get("FinalAngle", ""),
                    trial_data.get("CorrectAngle", ""),
                    trial_data.get("DifferenceAngle", ""),
                    trial_data.get("ErrorMeasure", "")
                ])
            else:
                output.extend([""] * 6)

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

def get_column_headers(total_pi_trials, total_pointing_judgements, total_pointing_tasks, total_pt_trials):
    headers = [
        "Player_Name", "RotationTime", "MovementTime", "CircuitTime",
        "HomingTime_1", "HomingTime_2", "TotalHomingTime", "TotalTrainingTime"
    ]

    for i in range(total_pi_trials):
        headers.extend([f"PI_TotalTime_{i}", f"PI_Distance_{i}", f"PI_DistRatio_{i}",
                        f"PI_FinalAngle_{i}", f"PI_Angle_{i}", f"Corrected_PI_Angle_{i}"])

    for i in range(total_pointing_tasks):
        for j in range(total_pointing_judgements):
            headers.append(f"PointingJudgement_AbsoluteError_{i}_Trial_{j}")

    headers.extend([
        "PointingJudgementTotalTime", "MapTotalTime", "CalculatedMapTotalTimeSeconds", "MapRSq",
        "MemoryTotalTime", "CalculatedMemoryTotalTimeSeconds", "MemoryPercentCorrect",
        "Overall_PerpectiveIdleTime", "Overall_PerspectiveTotalTime", "Overall_PerspectiveErrorMeasure",
        "SPACEStartTime", "SPACEEndTime", "SPACETotalTime"
    ])

    # Add headers for map coordinates
    landmarks = ["Nest", "Cave", "Arch", "Tree", "Volcano", "Waterfall"]
    for landmark in landmarks:
        headers.extend([f"{landmark}_X", f"{landmark}_Y"])

    for i in range(total_pt_trials):
        headers.extend([f"PerspectiveTotalTime_{i}", f"PerpectiveIdleTime_{i}",
                        f"PerpectiveFinalAngle_{i}", f"PerpectiveCorrectAngle_{i}",
                        f"PerpectiveDifferenceAngle_{i}", f"PerspectiveErrorMeasure_{i}"])

    return headers

def JSONtoCSV(json_files, csv_filename, total_pi_trials, total_pointing_judgements, total_pointing_tasks, total_pt_trials):
    logger.info(f"Processing {len(json_files)} JSON files")
    
    processor = JSONProcessor(total_pi_trials, total_pointing_judgements, total_pointing_tasks, total_pt_trials)
    headers = get_column_headers(total_pi_trials, total_pointing_judgements, total_pointing_tasks, total_pt_trials)
    
    data = [processor.process_file(f) for f in json_files if f is not None]
    data = [row for row in data if row is not None]
    
    df = pd.DataFrame(data, columns=headers)
    logger.info(f"DataFrame shape: {df.shape}")
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    
    return df

def get_column_groups(df, total_pi_trials, total_pointing_judgements, total_pointing_tasks, total_pt_trials):
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
            "PI_averages": {
                "PI TotalTime":["Avg_PI_TotalTime"],
                "PI Distance": ["Avg_PI_Distance"],
                "PI DistanceRatio":["Avg_PI_DistRatio"],
                "PI FinalAngle": ["Avg_PI_FinalAngle"],
                "Corrected PI Angle": ["Avg_Corrected_PI_Angle"]
            }
        },
        "Pointing error": {
            "Error (every trial)": [
                f"Avg_PointingJudgement_AbsoluteError_{i}" for i in range(total_pi_trials)
            ],
            "Pointing_Error_Average_all":
                [
                    "Average_PointingJudgementError_all"
                ]
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
            "Perspective_Taking_Time": [
                "Overall_PerspectiveTotalTime"
            ],
            "Perspective_Error_Average": [
                "Avg_PerspectiveErrorMeasure"
            ], 
        },
        "Overall Measures": [
            'SPACEStartTime', 'SPACEEndTime', 'SPACETotalTime'
        ]
    }

    # Group PI trial columns
    for i in range(total_pi_trials):
        pi_cols = [
            f'PI_TotalTime_{i}', f'PI_Distance_{i}', f'PI_DistRatio_{i}',
            f'PI_FinalAngle_{i}', f'Corrected_PI_Angle_{i}'
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
            "PI_averages": [
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
            "Perspective Error Average": [
                "Avg_PerspectiveErrorMeasure"
            ]
        },
        "Overall Measures": [
            'SPACEStartTime', 'SPACEEndTime', 'SPACETotalTime'
        ]
        }
    return column_groups