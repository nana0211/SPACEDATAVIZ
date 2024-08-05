import json
import csv
import glob, os
import logging
import pandas as pd
from datetime import datetime
from dateutil import parser
from getTrialNumbers import findEstimatedLandmarks

# for every PointingJudgement in every PointingTask
def GetPointingJudgement_AbsoluteError(idata, i, j):
    if len(idata["Sessions"]["Egocentric"]) < 1:
        return ""
    temp = idata["Sessions"]["Egocentric"][0]["PointingTasks"]
    # if index i or j are out of bounds just return empty string
    if i >= len(temp):
        return ""
    temp = idata["Sessions"]["Egocentric"][0]["PointingTasks"][i]["PointingJudgements"]
    if j >= len(temp):
        return ""
    return idata["Sessions"]["Egocentric"][0]["PointingTasks"][i]["PointingJudgements"][j]["Absolute_Error"]
def GetPointingJudgement_CorrectAngle(idata, i, j):
    if len(idata["Sessions"]["Egocentric"]) < 1:
        return ""
    temp = idata["Sessions"]["Egocentric"][0]["PointingTasks"]
    # if index i or j are out of bounds just return empty string
    if i >= len(temp):
        return ""
    temp = idata["Sessions"]["Egocentric"][0]["PointingTasks"][i]["PointingJudgements"]
    if j >= len(temp):
        return ""
    return idata["Sessions"]["Egocentric"][0]["PointingTasks"][i]["PointingJudgements"][j]["Correct_Angle"]
def GetPointingJudgement_EstimatedAngle(idata, i, j):
    if len(idata["Sessions"]["Egocentric"]) < 1:
        return ""
    temp = idata["Sessions"]["Egocentric"][0]["PointingTasks"]
    # if index i or j are out of bounds just return empty string
    if i >= len(temp):
        return ""
    temp = idata["Sessions"]["Egocentric"][0]["PointingTasks"][i]["PointingJudgements"]
    if j >= len(temp):
        return ""
    return idata["Sessions"]["Egocentric"][0]["PointingTasks"][i]["PointingJudgements"][j]["Estimated_Angle"]
def GetPointingJudgement_RawError(idata, i, j):
    if len(idata["Sessions"]["Egocentric"]) < 1:
        return ""
    temp = idata["Sessions"]["Egocentric"][0]["PointingTasks"]
    # if index i or j are out of bounds just return empty string
    if i >= len(temp):
        return ""
    temp = idata["Sessions"]["Egocentric"][0]["PointingTasks"][i]["PointingJudgements"]
    if j >= len(temp):
        return ""
    return idata["Sessions"]["Egocentric"][0]["PointingTasks"][i]["PointingJudgements"][j]["Raw_Error"]
def GetPointingJudgement_CAMinusRE(idata,i,j):
    if len(idata["Sessions"]["Egocentric"]) < 1:
        return ""
    temp = idata["Sessions"]["Egocentric"][0]["PointingTasks"]
    # if index i or j are out of bounds just return empty string
    if i >= len(temp):
        return ""
    temp = idata["Sessions"]["Egocentric"][0]["PointingTasks"][i]["PointingJudgements"]
    if j >= len(temp):
        return ""
    cor_ang = float(idata["Sessions"]["Egocentric"][0]["PointingTasks"][i]["PointingJudgements"][j]["Correct_Angle"])
    est_ang = float(idata["Sessions"]["Egocentric"][0]["PointingTasks"][i]["PointingJudgements"][j]["Estimated_Angle"])
    return  cor_ang - est_ang 
def GetPointingJudgementTotalTime(idata):
    if len(idata["Sessions"]["Egocentric"]) < 1:
        return ""
    if len(idata["Sessions"]["Egocentric"][0]["PointingTasks"]) < 1:
        return ""
    lastPointingTask = len(idata["Sessions"]["Egocentric"][0]["PointingTasks"]) - 1
    if len(idata["Sessions"]["Egocentric"][0]["PointingTasks"][lastPointingTask]) < 1:
        return ""
    lastPointingJudgement = len(idata["Sessions"]["Egocentric"][0]["PointingTasks"][lastPointingTask]["PointingJudgements"]) - 1
    if len(idata["Sessions"]["Egocentric"][0]["PointingTasks"][lastPointingTask]) < 1:
        return ""
    lastRotation = len(idata["Sessions"]["Egocentric"][0]["PointingTasks"][lastPointingTask]["PointingJudgements"][lastPointingJudgement]["rawData"]["Rotations"]) - 1
    timeStart = parser.parse(idata["Sessions"]["Egocentric"][0]["PointingTasks"][0]["PointingJudgements"][0]["rawData"]["Rotations"][0]["timeStamp"])
    timeEnd = parser.parse(idata["Sessions"]["Egocentric"][0]["PointingTasks"][lastPointingTask]["PointingJudgements"][lastPointingJudgement]["rawData"]["Rotations"][lastRotation]["timeStamp"])
    return (timeEnd - timeStart).total_seconds()

# Get specific data from Path Integration
def GetPI_TotalTime(jsonData):
    return jsonData["totalTime"]
def GetPI_Distance(jsonData):
    return jsonData["PIDistance"]
def GetPI_DistRatio(jsonData):
    return jsonData["PIDistanceRatio"]
def GetPI_FinalAngle(jsonData):
    return jsonData["FinalPIAngle"]
def GetPIAngle(jsonData):
    return jsonData["PIAngle"]
def GetCorrectedPIAngle(jsonData):
    return jsonData["CorrectedPIAngle"]

# Get specific data from Perspective Taking
def GetPT_TotalTime(jsonData):
    return jsonData["TotalTime"]
def GetPT_IdleTime(jsonData):
    return jsonData["TotalIdleTime"]
def GetPT_FinalAngle(jsonData):
    return jsonData["FinalAngle"]
def GetPT_CorrectAngle(jsonData):
    return jsonData["CorrectAngle"]
def GetPT_DifferenceAngle(jsonData):
    return jsonData["DifferenceAngle"]
def GetPT_ErrorMeasure(jsonData):
    return jsonData["ErrorMeasure"]

# Functions to get simple items from the json
# (single items that are not from a list/array)
def GetPlayerName(idata):
    return idata["MetaData"]["Player_Name"]
def GetRotationTime(idata):
    if "Training" in idata.keys():
        return idata["Training"]["phase1"]["totalTime"]
    else:
        return ""
def GetCircuitTime(idata):
    if "Training" in idata.keys():
        is_phase3_null = idata.get('Training', {}).get('phase3') is None
        if is_phase3_null == False:
           return idata["Training"]["phase3"]["totalTime"]
        else:
           return ""
def GetMovementTime(idata):
    if "Training" in idata.keys():
        return idata["Training"]["phase2"]["totalTime"]
    else:
        return ""

def GetHomingTime1(idata):
    if "Training" in idata.keys():
        return idata["Training"]["phase5"]["Trials"][0]["Data"]["totalTime"]
    else:
        return ""

def GetHomingTime2(idata):
    if "Training" in idata.keys():
        return idata["Training"]["phase5"]["Trials"][1]["Data"]["totalTime"]
    else:
        return ""

def GetHomingTime3(idata):
    if "Training" in idata.keys():
            return idata["Training"]["phase5"]["Trials"][1]["Data"]["totalTime"] + idata["Training"]["phase5"]["Trials"][0]["Data"]["totalTime"]
    else:
        return ""

def GetTotalTrainingTime(idata):
    if "Training" in idata.keys():
            return GetHomingTime3(idata) + GetMovementTime(idata) + GetRotationTime(idata)
    else:
        return ""


########################################################################
def GetMapTotalTime(idata):
    if len(idata["Sessions"]["Mapping"]) < 1:
        return ""
    return idata["Sessions"]["Mapping"][0]["TotalTime"]
def GetCalculatedMapTotalTime(idata):
    if len(idata["Sessions"]["Mapping"]) < 1:
        return ""
    timeStart = parser.parse(idata["Sessions"]["Mapping"][0]["StartTimeStamp"])
    timeEnd = parser.parse(idata["Sessions"]["Mapping"][0]["EndTimeStamp"])
    return (timeEnd - timeStart)
def GetCalculatedMapTotalTimeSeconds(idata):
    if len(idata["Sessions"]["Mapping"]) < 1:
        return ""
    timeStart = parser.parse(idata["Sessions"]["Mapping"][0]["StartTimeStamp"])
    timeEnd = parser.parse(idata["Sessions"]["Mapping"][0]["EndTimeStamp"])
    return (timeEnd - timeStart).total_seconds()
def GetMapRSq(idata):
    if len(idata["Sessions"]["Mapping"]) < 1:
        return ""
    return idata["Sessions"]["Mapping"][0]["BidimensionalRegression"]["Euclidean"]["R2"]
def GetMapCoordinate(idata):
    if len(idata["Sessions"]["Mapping"]) < 1:
        return ""
    return idata["Sessions"]["Mapping"][0]["EstimatedCoordinates"]
def GetMapCoordinateXY(idata):
    xy_data = GetMapCoordinate(idata)
    map_data =[]
    if "Nest" in xy_data.keys():
        map_data.extend([xy_data["Nest"]["X"],xy_data["Nest"]["Y"]])
    if "Cave" in xy_data.keys():
        map_data.extend([xy_data["Cave"]["X"],xy_data["Cave"]["Y"]])
    if "Arch" in xy_data.keys():
        map_data.extend([xy_data["Arch"]["X"],xy_data["Arch"]["Y"]])
    if "Tree" in xy_data.keys():
        map_data.extend([xy_data["Tree"]["X"],xy_data["Tree"]["Y"]])
    if "Volcano" in xy_data.keys():
        map_data.extend([xy_data["Volcano"]["X"],xy_data["Volcano"]["Y"]])
    if "Waterfall" in xy_data.keys():
        map_data.extend([xy_data["Waterfall"]["X"],xy_data["Waterfall"]["Y"]])
    return map_data
####################### Memory ##############################################
def GetMemoryTotalTime(idata):
    if len(idata["Sessions"]["Memory"]) < 1:
        return ""
    return idata["Sessions"]["Memory"][0]["TotalTime"]
def GetCalculatedMemoryTotalTime(idata):
    if len(idata["Sessions"]["Memory"]) < 1:
        return ""
    timeStart = parser.parse(idata["Sessions"]["Memory"][0]["StartTimeStamp"])
    timeEnd = parser.parse(idata["Sessions"]["Memory"][0]["EndTimeStamp"])
    return (timeEnd - timeStart)
def GetCalculatedMemoryTotalTimeSeconds(idata):
    if len(idata["Sessions"]["Memory"]) < 1:
        return ""
    timeStart = parser.parse(idata["Sessions"]["Memory"][0]["StartTimeStamp"])
    timeEnd = parser.parse(idata["Sessions"]["Memory"][0]["EndTimeStamp"])
    return (timeEnd - timeStart).total_seconds()
def GetMemoryPercentCorr(idata):
    if len(idata["Sessions"]["Memory"]) < 1:
        return ""
    return idata["Sessions"]["Memory"][0]["PercentCorrect"]
def GetPerspectiveTotalIdleTime(idata):
    if len(idata["Sessions"]["PerspectiveTaking"]) < 1:
        return ""
    return idata["Sessions"]["PerspectiveTaking"][0]["TotalIdleTime"]

def GetPerspectiveTotalTime(idata):
    if len(idata["Sessions"]["PerspectiveTaking"]) < 1:
        return ""
    return idata["Sessions"]["PerspectiveTaking"][0]["TotalTime"]

def GetPerspectiveError(idata):
    if len(idata["Sessions"]["PerspectiveTaking"]) < 1:
        return ""
    return idata["Sessions"]["PerspectiveTaking"][0]["AverageErrorMeasure"]

def GetCalculatedSPACEStartTime(idata):
    return idata["MetaData"]["Start_Timestamp"]

def GetCalculatedSPACEEndTime(idata):
    return idata["MetaData"]["End_Timestamp"]

def GetCalculatedSPACETotalTimeSeconds(idata):
    if(idata["MetaData"]["Start_Timestamp"] and idata["MetaData"]["End_Timestamp"]):
        timeStart = parser.parse(idata["MetaData"]["Start_Timestamp"])
        timeEnd = parser.parse(idata["MetaData"]["End_Timestamp"])
        return (timeEnd - timeStart).total_seconds()
    else:
        return ""
#endregion


def GetData_with_averages(files, totalPathIntegrationTrials, totalPointingJudgements, totalPointingTasks, totalPerspectiveTakingTrials):
    logging.info(f"GetData called with {len(files)} files")
    outputArray = []
    
    # Define column headers
    columnsBefore = ["Player_ID", "RotationTime", "MovementTime", "CircuitTime",
                     "TotalHomingTime", "TotalTrainingTime"]

    # Add Columns for Path Integration trial data
    for i in range(totalPathIntegrationTrials):
        columnsBefore.extend([f"PI_TotalTime_{i}", f"PI_Distance_{i}", f"PI_DistRatio_{i}",
                              f"PI_FinalAngle_{i}", f"Corrected_PI_Angle_{i}"])
    
    # Add columns for Pointing Judgement tasks
    for i in range(totalPointingTasks):
        for j in range(totalPointingJudgements):
            columnsBefore.append(f"PointingJudgement_AbsoluteError_{i}_Trial_{j}")
    
    
    columnsAfter = ["Average_PointingJudgementError_all","PointingJudgementTotalTime", "MapTotalTime", "MapRSq"]
    
    # Add landmark columns
    landmarks = findEstimatedLandmarks(files[0])
    columnsAfter.extend(landmarks)
    
    columnsAfter.extend(["MemoryTotalTime", "MemoryPercentCorrect", "Overall_PerspectiveTotalTime", 
                         "SPACEStartTime", "SPACEEndTime", "SPACETotalTime"])

    # Add columns for Perspective Taking tasks
    for i in range(totalPerspectiveTakingTrials):
        columnsAfter.append(f"PerspectiveErrorMeasure_{i}")

    # Add columns for average PI trials, Pointing Judgements, and Perspective Taking
    avg_pi_columns = ["Avg_PI_TotalTime", "Avg_PI_Distance", "Avg_PI_DistRatio", "Avg_PI_FinalAngle", "Avg_Corrected_PI_Angle"]
    avg_pointing_columns = [f"Avg_PointingJudgement_AbsoluteError_{i}" for i in range(totalPointingTasks)]
    avg_perspective_columns = ["Avg_PerspectiveErrorMeasure"]
    
    columnsBefore.extend(columnsAfter)
    columnsBefore.extend(avg_pi_columns + avg_pointing_columns + avg_perspective_columns)

    outputArray.append(columnsBefore)
    logging.info(f"Column headers created. Total columns: {len(columnsBefore)}")

    # For every json file found
    for f in files:
        try:
            logging.info(f"Processing file: {f}")
            outputLine = []
            with open(f, 'r') as file:
                data = json.load(file)
 
            # Get first set of simple data
            outputLine.extend([GetPlayerName(data), GetRotationTime(data), GetMovementTime(data), GetCircuitTime(data),
                               GetHomingTime3(data), GetTotalTrainingTime(data)])

            pi_totals = []
            pi_distances = []
            pi_dist_ratios = []
            pi_final_angles = []
            pi_corrected_angles = []

            # Get Path Integration Data
            if "Sessions" in data and "PathIntegration" in data["Sessions"] and len(data["Sessions"]["PathIntegration"]) > 0:
                for i in range(totalPathIntegrationTrials):
                    temp = data["Sessions"]["PathIntegration"][0]["Trials"]
                    if i >= len(temp):
                        outputLine.extend([""] * 5)
                    else:
                        pit_data = data["Sessions"]["PathIntegration"][0]["Trials"][i]["Data"]
                        total_time = GetPI_TotalTime(pit_data)
                        distance = GetPI_Distance(pit_data)
                        dist_ratio = GetPI_DistRatio(pit_data)
                        final_angle = GetPI_FinalAngle(pit_data)
                        corrected_angle = GetCorrectedPIAngle(pit_data)

                        outputLine.extend([total_time, distance, dist_ratio, final_angle, corrected_angle])

                        pi_totals.append(total_time)
                        pi_distances.append(distance)
                        pi_dist_ratios.append(dist_ratio)
                        pi_final_angles.append(final_angle)
                        pi_corrected_angles.append(corrected_angle)
            else:
                outputLine.extend([""] * (5 * totalPathIntegrationTrials))

            # Get Pointing Judgements Data
            pointing_errors = [[] for _ in range(totalPointingTasks)]
            for i in range(totalPointingTasks):
                for j in range(totalPointingJudgements):
                    error = GetPointingJudgement_AbsoluteError(data, i, j)
                    outputLine.append(error)
                    if error != "":
                        pointing_errors[i].append(error)
            #Get Average pointing error:
            
            outputLine.extend([sum(sum(pointing_errors[totalPointingTasks][totalPointingJudgements] for totalPointingJudgements in range(len(pointing_errors[totalPointingTasks]))) / len(pointing_errors[totalPointingTasks]) for totalPointingTasks in range(len(pointing_errors))) / len(pointing_errors)])
            
            # Get Remaining set of simple data
            outputLine.extend([
                GetPointingJudgementTotalTime(data),
                GetMapTotalTime(data),
                GetMapRSq(data)
            ])
            outputLine.extend(GetMapCoordinateXY(data))
            outputLine.extend([
                GetMemoryTotalTime(data),
                GetMemoryPercentCorr(data),
                GetPerspectiveTotalTime(data),
                GetCalculatedSPACEStartTime(data),
                GetCalculatedSPACEEndTime(data),
                GetCalculatedSPACETotalTimeSeconds(data)
            ])

            # Get Perspective Taking Data
            perspective_errors = []
            if "Sessions" in data and "PerspectiveTaking" in data["Sessions"] and len(data["Sessions"]["PerspectiveTaking"]) > 0:
                for i in range(totalPerspectiveTakingTrials):
                    temp = data["Sessions"]["PerspectiveTaking"][0]["Trials"]
                    if i >= len(temp):
                        outputLine.extend([""] * 1)
                    else:
                        ptt_data = temp[i]
                        error_measure = GetPT_ErrorMeasure(ptt_data)
                        outputLine.extend([error_measure])
                        if error_measure != "":
                            perspective_errors.append(error_measure)
            else:
                outputLine.extend([""] * totalPerspectiveTakingTrials)

            # Calculate and add average PI values
            avg_pi_values = [
                sum(pi_totals) / len(pi_totals) if pi_totals else "",
                sum(pi_distances) / len(pi_distances) if pi_distances else "",
                sum(pi_dist_ratios) / len(pi_dist_ratios) if pi_dist_ratios else "",
                sum(pi_final_angles) / len(pi_final_angles) if pi_final_angles else "",
                sum(pi_corrected_angles) / len(pi_corrected_angles) if pi_corrected_angles else ""
            ]

            # Calculate and add average Pointing Judgements errors
            avg_pointing_errors = [
                sum(pointing_errors[i]) / len(pointing_errors[i]) if pointing_errors[i] else ""
                for i in range(totalPointingTasks)
            ]

            # Calculate and add average Perspective Taking errors
            avg_perspective_error = sum(perspective_errors) / len(perspective_errors) if perspective_errors else ""

            # Append averages to the output line
            outputLine.extend(avg_pi_values)
            outputLine.extend(avg_pointing_errors)
            outputLine.append(avg_perspective_error)

            # Add participants data to final output
            outputArray.append(outputLine)
            logging.info(f"Processed file: {f}, outputLine length: {len(outputLine)}")

        except Exception as e:
            logging.error(f"Error processing file {f}: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())

    logging.info(f"GetData finished, outputArray has {len(outputArray)} rows")
    return outputArray

def JSONtoCSV_with_average(json_files, csv_filename, totalPathIntegrationTrials,totalPointingJudgements,totalPointingTasks,totalPerspectiveTakingTrials):
    print(f"Processing {len(json_files)} JSON files")
    dataArray = GetData_with_averages(json_files,totalPathIntegrationTrials,totalPointingJudgements,totalPointingTasks,totalPerspectiveTakingTrials)
    print(f"Data array shape: {len(dataArray)} rows, {len(dataArray[0])} columns")
    df = pd.DataFrame(dataArray[1:], columns=dataArray[0])
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    return df

def get_column_groups_full_with_average(df, totalPathIntegrationTrials, totalPointingJudgements, totalPointingTasks, totalPerspectiveTakingTrials):
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
                f"Avg_PointingJudgement_AbsoluteError_{i}" for i in range(totalPointingTasks)
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
    for i in range(totalPathIntegrationTrials):
        pi_cols = [
            f'PI_TotalTime_{i}', f'PI_Distance_{i}', f'PI_DistRatio_{i}',
            f'PI_FinalAngle_{i}', f'Corrected_PI_Angle_{i}'
        ]
        if any(col in df.columns for col in pi_cols):
            if isinstance(column_groups["PI (for each trial)"], dict):
                column_groups["PI (for each trial)"][f'PI_trial_{i}'] = pi_cols

    # Group Pointing error columns
    for i in range(totalPointingTasks):
        pointing_cols = [f'PointingJudgement_AbsoluteError_{i}_Trial_{j}' for j in range(totalPointingJudgements)]
        if any(col in df.columns for col in pointing_cols):
            if isinstance(column_groups["Pointing error"], dict):
                column_groups["Pointing error"][f'Pointing_trial_{i}'] = pointing_cols

    # Group Perspective taking columns
    for i in range(totalPerspectiveTakingTrials):
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
    
    
