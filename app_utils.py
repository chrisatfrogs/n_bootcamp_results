import os
import json
import pandas as pd
from markdown2 import Markdown
from turn_class import create_session, Turn

# Convert markdown to HTML
def convert_markdown_to_html(markdown: str) -> str:
    '''
    Converts markdown text to HTML

    Parameters
    ----------
    markdown : str
        Markdown text

    Returns
    -------
    html : str
        HTML text
    '''
    html = Markdown().convert(markdown)
    # Return HTML
    return html

# Dictionary for list values
def create_listvalues(
    lv1: str = None,
    lv2: str = None,
    lv3: str = None,
    lv4: str = None,
    lv5: str = None,
    lv6: str = None,
    model1: str = None,
    model2: str = None,
    model3: str = None,
    model4: str = None,
    model5: str = None,
    model6: str = None):

    listvalues = {}
    for lv, model in zip([lv1, lv2, lv3, lv4, lv5, lv6], [model1, model2, model3, model4, model5, model6]):
        if lv and lv != '' and model:
            listvalues[str(float(lv))] = model

    return listvalues



## Function to create turn subfolder from turn name
def create_turn_subfolder(turn_name: str):
    '''
    Creates a turn subfolder from a turn name
    '''
    cwd = os.getcwd()
    turn_folder = os.path.join(cwd, 'turns')
    # Create turn subfolder
    turn_subfolder = os.path.join(turn_folder, turn_name)
    # Check if turn subfolder exists
    if not os.path.exists(turn_subfolder):
        # Create turn subfolder
        os.makedirs(turn_subfolder)
    # Return turn subfolder
    return turn_subfolder

# Create model.json with turn type and research findings
def create_model_json(turn_subfolder: os.PathLike, list_dict: dict, turn_format: str, turn_type: str, research_objectives: str, research_findings: str):
    '''
    Creates model.json with turn type and research findings
    '''
    model_json = os.path.join(turn_subfolder, 'model.json')
    # Check if model.json exists
    if not os.path.exists(model_json):
        # Create model.json
        with open(model_json, 'w') as f:
            json_data = {
                'type': turn_type,
                'turn_format': turn_format,
                'models': list_dict,
                'research_objectives': convert_markdown_to_html(research_objectives),
                'research_findings': convert_markdown_to_html(research_findings)
            }

            json_object = json.dumps(json_data, indent=4)
            f.write(json_object)
    # Return model.json
    return model_json

# Copy uploaded CSV file and rename it to dataset.csv
def copy_uploaded_file(turn_subfolder: str, uploaded_file):
    '''
    Copies uploaded CSV file and renames it to dataset.csv

    Parameters
    ----------
    turn_subfolder : str
        Turn subfolder
    uploaded_file : file
        Uploaded CSV file

    Returns
    -------
    dataset_csv : str
        Path to dataset.csv
    
    '''
    # Create dataset.csv
    dataset_csv = os.path.join(turn_subfolder, 'dataset.csv')
    # Check if dataset.csv exists
    if not os.path.exists(dataset_csv):
        # Copy uploaded file to dataset.csv
        df = pd.read_csv(uploaded_file)
        df.to_csv(dataset_csv, index=False)
    # Return dataset.csv
    return dataset_csv

def create_turn(turn_name: str, 
    turn_type: str,
    turn_format: str,
    turn_uuid: str,
    turn_description: str,
    turn_dataset,
    research_objectives: str,
    research_findings: str,
    listvalues: dict,
    turn_texts = None):
    '''
    Creates a turn

    Parameters
    ----------
    turn_name : str
        Turn name
    turn_type : str
        Turn type
    uploaded_file : file
        Uploaded CSV file
    research_findings : str
        Research findings

    Returns
    -------
    turn : Turn
        Turn object
    '''

    # Add turn to db
    session = create_session()
    turn = Turn(turn_name=turn_name, turn_uuid=turn_uuid, turn_description=turn_description)
    session.add(turn)
    session.commit()

    # Create turn subfolder
    turn_subfolder = create_turn_subfolder(turn_name)
    
    # Create model.json
    create_model_json(turn_subfolder, listvalues, turn_format, turn_type, research_objectives, research_findings)

    # Copy uploaded file
    copy_uploaded_file(turn_subfolder, turn_dataset)

    return turn
