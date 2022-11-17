import json
import uuid
import yaml
import sqlalchemy
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
from yaml import SafeLoader

import app_utils
from turn_class import create_session, engine, Turn
from turns.turnx103 import comments
from turn import OldNonFictionDataset, NewNonFictionDataset, HoroscopeDataset

## Create app_utils.py

# Set the page layout to wide
st.set_page_config(layout="wide", page_title="Turn results")
session = create_session()

if not sqlalchemy.inspect(engine).has_table('turns'):
    Turn.create_turns()

@st.cache(allow_output_mutation=True)
def load_data(turn: str) -> dict:
    """Returns the data for the given turn."""
    folder = f'turns/{turn}'
    with open(f'{folder}/model.json', 'r') as f:
        model = json.load(f)
    
    dataset = f'turns/{turn}/dataset.csv'
    turn_type = model['type']
    models = {
        float(key): value for key, value in model['models'].items()
    }
    turn_format = model['turn_format'] if 'turn_format' in model else 'old'
    research_objectives = model['research_objectives']
    research_findings = model['research_findings']

    return {turn: {
        'type': turn_type,
        'dataset': dataset,
        'models': models,
        'turn_format': turn_format,
        'research_objectives': research_objectives,
        'research_findings': research_findings
    }}

def create_block(title: str, column_selection: list = None, data: pd.DataFrame = None, chart = None, markdown: str = None):
    """Creates a block with the given title and text."""
    if title:
        st.markdown(f'## {title}')
        if data is not None:
            if column_selection is not None:
                st.dataframe(data[column_selection], use_container_width=True)
            else:
                st.dataframe(data, use_container_width=True)
        if chart is not None:
            st.plotly_chart(chart, use_container_width=True)
        if markdown is not None:
            st.markdown(markdown, unsafe_allow_html=True)
        st.markdown('<br />', unsafe_allow_html=True)

with open('config.yaml', 'r') as f:
    config = yaml.load(f, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    
    st.write(f'Welcome *{name}*!')
    with st.sidebar:
        st.markdown('# :tada: Turn result dashboard :tada:')
        
        turns = [turn[0] for turn in session.query(Turn.turn_name).order_by(Turn.turn_id).all()]
        turn = st.selectbox('Select a turn', turns, key='turn-selection')
        file_selector = None
        if username == "tqa":
            file_selector = st.checkbox('Add new turn', key='add-turn')
        st.markdown('<br/><br/>', unsafe_allow_html=True)
    
    authenticator.logout('Logout', 'sidebar')
    
    if not file_selector:
    
        data = load_data(turn)

        # Load attributes
        turn_type = data[turn]['type']
        turn_format = data[turn]['turn_format']
        dataset = data[turn]['dataset']
        models = data[turn]['models']
        research_objectives = data[turn]['research_objectives']
        research_findings = data[turn]['research_findings']
        
        if turn_type == 'non-fiction':
            if turn_format == 'old':
                turn_dataset = OldNonFictionDataset(turn=turn, data_source = dataset, models=models)
            else:
                turn_dataset = NewNonFictionDataset(turn=turn, data_source = dataset, models=models)
        elif turn_type == 'horoscope':
            turn_dataset = HoroscopeDataset(turn=turn, data_source = dataset, models=models)
        
        n_texts, n_users, n_ratings = turn_dataset.get_general_info()
        listvalue_table = turn_dataset.get_list_info()
        bar_chart = turn_dataset.get_bar_chart()
        box_plot = turn_dataset.get_box_plot()
        significance_table = turn_dataset.get_significance_table()

        st.title(f'{turn}')
        st.markdown('<br />', unsafe_allow_html=True)

        if turn == 'turnx106nf':
            st.markdown('<p style="font-size: 20px; font-weight: bold; color: red;">THIS IS AN ONGOING TURN</p>', unsafe_allow_html=True)

        # Get general information about the dataset
        st.markdown('## General information')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Number of texts", n_texts)

<<<<<<< Updated upstream
    # Get general information about the dataset
    st.markdown('## General information')
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of texts", n_texts)
=======
        with col2:
            st.metric("Number of users", n_users)

        with col3:
            st.metric("Number of ratings", n_ratings)
        
        st.markdown('<br/>', unsafe_allow_html=True)
>>>>>>> Stashed changes

        
        # Research objectives and List values section
        col4, col5 = st.columns([0.6, 0.4])
        with col4: 
            st.markdown('## Research objectives')
            st.markdown(research_objectives, unsafe_allow_html=True)

        with col5:
            st.markdown('## List values')
            st.dataframe(listvalue_table.style.hide(axis="index"), use_container_width=True)

        st.markdown('<br/>', unsafe_allow_html=True)
        
        # Research findings section
        create_block(title='Research findings', markdown=research_findings)

        if turn == 'turnx103':
            st.markdown('## Comment summary')
            st.markdown(comments.COMMENTS, unsafe_allow_html=True)

        # Mean summary section
        st.markdown('## Mean summary')
        mean_summary = turn_dataset.get_mean_table()
        mean_summary = mean_summary.rename(columns=models)
        column_selection = st.multiselect('Select models', mean_summary.columns)
        if column_selection:
            st.dataframe(mean_summary[column_selection].style.highlight_max(axis=1, color='#6493CC').format(precision=2), use_container_width=True, height=420)
        else:
            st.dataframe(mean_summary.style.highlight_max(axis=1, color='#6493CC').format(precision=2), use_container_width=True, height=420)
        st.markdown('<br/>', unsafe_allow_html=True)

        # Bar chart section
        create_block(title='Bar chart', chart=bar_chart)

        # Box plot section
        with st.expander('Show box plots'):
            create_block(title='Box plots', chart=box_plot)
                
        # Significance table section
        create_block(title='Significance table', markdown=significance_table)

        if turn_type == 'non-fiction' and turn_format == 'old':
            # Fact check section
            st.markdown('## Explicit and implicit fact checks')
            col10, col11 = st.columns(2)
            with col10:
                st.plotly_chart(turn_dataset.get_pie_chart('factual_correctness'), use_container_width=True)
                    
            with col11:
                st.plotly_chart(turn_dataset.get_pie_chart('implicit_fact_check'), use_container_width=True)
    
    else:
        st.markdown('# Add new turn')
        st.markdown('<br/>', unsafe_allow_html=True)
        with st.form(key='add-turn-form'):
            turn_id = st.text_input('Turn ID', value=uuid.uuid4(), disabled=True)
            turn_name = st.text_input('Turn name', value='turnx', key='turn-name')
            turn_type = st.selectbox('Turn type', ['non-fiction', 'horoscope'], key='turn-type')
            turn_format = st.selectbox('Turn format', ['old', 'new'], key='turn-format')
            turn_dataset = st.file_uploader('Turn dataset', key='turn-dataset', type=['csv', 'xlsx'])
            turn_texts = st.file_uploader('Turn texts', key='turn-texts', type=['csv', 'xlsx'])
            turn_description = st.text_area('Turn description', key='turn-description')
            with st.expander('List values'):
                col1, col2 = st.columns(2)
                with col1:
                    lv_1 = st.text_input('List value 1', key='lv-1')
                    lv_2 = st.text_input('List value 2', key='lv-2')
                    lv_3 = st.text_input('List value 3', key='lv-3')
                    lv_4 = st.text_input('List value 4', key='lv-4')
                    lv_5 = st.text_input('List value 5', key='lv-5')
                    lv_6 = st.text_input('List value 6', key='lv-6')
                with col2:
                    model_1 = st.text_input('Model 1', key='model-1')
                    model_2 = st.text_input('Model 2', key='model-2')
                    model_3 = st.text_input('Model 3', key='model-3')
                    model_4 = st.text_input('Model 4', key='model-4')
                    model_5 = st.text_input('Model 5', key='model-5')
                    model_6 = st.text_input('Model 6', key='model-6')
            turn_objectives = st.text_area('Research objectives', key='turn-objectives', help='You can use Markdown to format the text')
            turn_findings = st.text_area('Research findings', key='turn-findings', help='You can use Markdown to format the text')
            submit_button = st.form_submit_button(label='Submit')

        
        if submit_button:
            if turn_name is None or turn_name == '':
                st.error('Turn name is required')
            elif turn_type is None or turn_type == '':
                st.error('Turn type is required')
            elif turn_format is None or turn_format == '':
                st.error('Turn format is required')
            elif turn_dataset is None:
                st.error('Turn dataset is required')
            elif turn_objectives is None or turn_objectives == '':
                st.error('Research objectives are required')
            elif turn_findings is None or turn_findings == '':
                st.error('Research findings are required')
            

            listvalues = app_utils.create_listvalues(lv_1, lv_2, lv_3, lv_4, lv_5, lv_6, model_1, model_2, model_3, model_4, model_5, model_6)

            try:
                app_utils.create_turn(
                    turn_uuid=turn_id,
                    turn_name=turn_name,
                    turn_type=turn_type,
                    turn_format=turn_format,
                    turn_dataset=turn_dataset,
                    turn_texts=turn_texts,
                    turn_description=turn_description,
                    listvalues=listvalues,
                    research_objectives=turn_objectives,
                    research_findings=turn_findings
                )
                st.success('Turn added successfully')
            except Exception as error_message:
                st.error(error_message)

elif authentication_status == False:
    st.error('Username/password is incorrect')
