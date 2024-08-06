import os
import pandas as pd
import numpy as np
import zipfile
from flask import Flask, request, send_file, render_template, jsonify, send_from_directory, session
from werkzeug.utils import secure_filename
from finalJSONtoCSV import get_column_groups,JSONtoCSV,get_summary_columns
from getTrialNumbers import findAllTrials

app = Flask(__name__)
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = 'uploads'
DOWNLOAD_FOLDER = 'downloads'
ALLOWED_EXTENSIONS = {'json', 'zip'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clear_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/select_columns', methods=['POST'])
def select_columns():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    output_option = request.form.get('output-option')
    
    if file.filename == '':
        return 'No selected file', 400
    
    if file and allowed_file(file.filename):
        clear_folder(app.config['UPLOAD_FOLDER'])
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        # Store file path and output option in session
        session['file_path'] = file_path
        session['output_option'] = output_option
        return render_template('select_columns.html')
    return 'Invalid file type', 400

@app.route('/get_columns', methods=['POST'])
def get_columns():
    try:
        if request.method != 'POST':
            return jsonify({'error': 'Invalid request method'}), 405

        if request.content_type != 'application/json':
            return jsonify({'error': 'Invalid content type, expected application/json'}), 415

        output_option = request.json.get('output_option', 'all_trials')

        file_path = session.get('file_path')
        if not file_path:
            return jsonify({'error': 'No file uploaded'}), 400

        json_files = []
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(app.config['UPLOAD_FOLDER'])
                json_files = [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in zip_ref.namelist() if f.endswith('.json')]
        else:
            json_files = [file_path]

        num_pi, num_pj, num_pot, num_pet = findAllTrials(json_files[0])
        df = JSONtoCSV(json_files, os.path.basename(file_path), num_pi, num_pj, num_pot, num_pet)

        if output_option == 'average':
            column_groups = get_summary_columns()
        else:
            column_groups = get_column_groups(df, num_pi, num_pj, num_pot, num_pet)
            print("We get a full columns~~~~~~~")
        
        def process_group(group):
            if isinstance(group, dict):
                return {k: list(v.keys()) if isinstance(v, dict) else v for k, v in group.items()}
            return group

        top_level_groups = {k: process_group(v) for k, v in column_groups.items()}
        return jsonify(top_level_groups)
    
    except Exception as e:
        app.logger.error(f'Error in get_columns: {str(e)}')
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'error': 'An error occurred while fetching columns. Please try again.'}), 500
      
def expand_selected_columns(selected_columns, column_groups_all_trials, column_groups_average, df, output_option):
    expanded_columns = []

    if not selected_columns:
        return expanded_columns

    def add_columns(columns_to_add):
        if isinstance(columns_to_add, list):
            expanded_columns.extend(columns_to_add)
        else:
            expanded_columns.append(columns_to_add)

    def process_column_path(col_path, groups_dict):
        parts = col_path.split('.')
        current_dict = groups_dict
        for part in parts:
            if isinstance(current_dict, list):
                # If current_dict is a list, return it as we can't index it further
                return current_dict
            if part in current_dict:
                current_dict = current_dict[part]
            else:
                return None
        return current_dict

    if output_option == 'all_trials':
        for col in selected_columns:
            parts = col.split('.')
            if len(parts) == 1 and parts[0] == 'Player_ID':
                expanded_columns.append(parts[0])
            else:
                col_dict = process_column_path(col, column_groups_all_trials)
                if col_dict:
                    if isinstance(col_dict, list):
                        expanded_columns.extend(col_dict)
                    else:
                        expanded_columns.append(col_dict)
                elif col.startswith('PI_trial_'):
                    trial_num = col.split('_')[-1]
                    expanded_columns.extend([c for c in df.columns if c.startswith(f'PI_') and c.split('_')[2] == trial_num])
                    expanded_columns.append(f'Corrected_PI_Angle_{trial_num}')
                elif col.startswith('Pointing_trial_'):
                    trial_num = col.split('_')[-1]
                    expanded_columns.extend([c for c in df.columns if c.startswith(f'PointingJudgement_AbsoluteError_{trial_num}_')])
                elif col.startswith('Perspective_trial_'):
                    trial_num = col.split('_')[-1]
                    expanded_columns.extend([c for c in df.columns if c.startswith(f'Perspective') and f'_{trial_num}' in c])

    elif output_option == 'average':
        for col in selected_columns:
            parts = col.split('.')
            if len(parts) == 1 and parts[0] == 'Player_ID':
                expanded_columns.append(parts[0])
            else:
                col_dict = process_column_path(col, column_groups_average)
                if col_dict:
                    if isinstance(col_dict, list):
                        expanded_columns.extend(col_dict)
                    else:
                        expanded_columns.append(col_dict)

    return expanded_columns

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file_path = session.get('file_path')
        output_option = request.form.get('output_option', 'all_trials')
        if not file_path:
            return jsonify({'error': 'No file uploaded'}), 400

        selected_columns = request.form.getlist('columns')
        app.logger.info(f"Received selected columns: {selected_columns}")

        json_files = []
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(app.config['UPLOAD_FOLDER'])
                json_files = [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in zip_ref.namelist() if f.endswith('.json')]
        else:
            json_files = [file_path]

        csv_filename = 'combined_output.csv'
        csv_path = os.path.join(app.config['DOWNLOAD_FOLDER'], csv_filename)

        app.logger.info(f"Processing files: {json_files}")
        app.logger.info(f"Selected columns: {selected_columns}")

        num_pi, num_pj, num_pot, num_pet = findAllTrials(json_files[0])
        df = JSONtoCSV(json_files, os.path.basename(file_path), num_pi, num_pj, num_pot, num_pet)

        app.logger.info(f"DataFrame shape: {df.shape}")
        app.logger.info(f"DataFrame columns: {df.columns.tolist()}")

        column_groups_all_trials = get_column_groups(df, num_pi, num_pj, num_pot, num_pet)
        column_groups_average = get_summary_columns()

        expanded_columns = expand_selected_columns(selected_columns, column_groups_all_trials, column_groups_average, df, output_option)

        app.logger.info(f"Expanded columns: {expanded_columns}")
        existing_columns = list(dict.fromkeys([col for col in expanded_columns if col in df.columns]))
        app.logger.info(f"Existing columns: {existing_columns}")

        # Identify missing columns
        missing_columns = [col for col in expanded_columns if col not in df.columns]
        app.logger.info(f"Missing columns: {missing_columns}")
        if not existing_columns:
            return jsonify({'error': 'None of the selected columns were found in the data'}), 400

        new_df = df[existing_columns]

        app.logger.info(f"Final DataFrame shape: {new_df.shape}")
        app.logger.info(f"Final DataFrame columns: {new_df.columns.tolist()}")
        new_df.to_csv(csv_path, index=False)
        app.logger.info(f"CSV saved to: {csv_path}")
        return jsonify({'download_link': csv_filename})
    except Exception as e:
        app.logger.error(f'Error in upload_file: {str(e)}')
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({'error': 'An error occurred while processing the file. Please try again.'}), 500
    
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['DOWNLOAD_FOLDER'], filename), as_attachment=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7089, debug=True)