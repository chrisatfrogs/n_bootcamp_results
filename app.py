import json
import yaml
import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth
from yaml import SafeLoader

from turn import TurnDataset
from constants import TURNS



# Set the page layout to wide
st.set_page_config(layout="wide", page_title="Turn results")


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
    research_objectives = model['research_objectives']
    research_findings = model['research_findings']

    return {turn: {
        'type': turn_type,
        'dataset': pd.read_csv(dataset),
        'models': models,
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
        turn = st.selectbox('Select a turn', TURNS, key='turn-selection')
        st.markdown('<br/><br/>', unsafe_allow_html=True)
    
    data = load_data(turn)

    # Load attributes
    turn_type = data[turn]['type']
    dataset = data[turn]['dataset']
    models = data[turn]['models']
    research_objectives = data[turn]['research_objectives']
    research_findings = data[turn]['research_findings']

    # Load dataset and charts
    turn_dataset = TurnDataset(dataset, turn, models, turn_type)
    n_texts, n_users, n_ratings = turn_dataset.get_general_info()
    listvalue_df = turn_dataset.df[turn_dataset.df['model'].isin(turn_dataset.models.values())]
    bar_chart = turn_dataset.plotly_bar_chart()
    box_plot = turn_dataset.plotly_box_plot()
    significance_table = turn_dataset.generate_significance_table()

    st.title(f'{turn}')
    st.markdown('<br />', unsafe_allow_html=True)

    # Get general information about the dataset
    st.markdown('## General information')
    n_texts, n_users, n_ratings = turn_dataset.get_general_info()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of texts", n_texts)

    with col2:
        st.metric("Number of users", n_users)

    with col3:
        st.metric("Number of ratings", n_ratings)
    
    st.markdown('<br/>', unsafe_allow_html=True)

    
    # Research objectives and List values section
    col4, col5 = st.columns([0.6, 0.4])
    with col4: 
        st.markdown('## Research objectives')
        st.markdown(research_objectives, unsafe_allow_html=True)

    with col5:
        st.markdown('## List values')

        # Get the list values
        listvalue_df = turn_dataset.df[turn_dataset.df['model'].isin(turn_dataset.models.values())]
        st.dataframe(turn_dataset.generate_plotly_table(turn_dataset.models, listvalue_df).style.hide_index(), use_container_width=True)

    st.markdown('<br/>', unsafe_allow_html=True)
    
    # Research findings section
    create_block(title='Research findings', markdown=research_findings)

    # Mean summary section
    mean_summary = turn_dataset.eval_frame().T
    mean_summary = mean_summary.rename(columns=models)
    column_selection = st.multiselect('Select models', mean_summary.columns)
    if column_selection:
        st.dataframe(mean_summary[column_selection].style.highlight_max(axis=1, color='#6493CC').format(precision=2), use_container_width=True)
    else:
        st.dataframe(mean_summary.style.highlight_max(axis=1, color='#6493CC').format(precision=2), use_container_width=True)
    st.markdown('<br/>', unsafe_allow_html=True)

    # Bar chart section
    create_block(title='Bar chart', chart=bar_chart)

    # Box plot section
    with st.expander('Show box plots'):
        create_block(title='Box plots', chart=box_plot)
            
    # Significance table section
    create_block(title='Significance table', markdown=significance_table)

    if turn_type == 'non-fiction':
        # Fact check section
        st.markdown('## Explicit and implicit fact checks')
        col10, col11 = st.columns(2)
        with col10:
            st.plotly_chart(turn_dataset.plotly_pie_chart('factual_correctness'), use_container_width=True)
                
        with col11:
            st.plotly_chart(turn_dataset.plotly_pie_chart('implicit_fact_check'), use_container_width=True)
    
    authenticator.logout('Logout', 'sidebar')

elif authentication_status == False:
    st.error('Username/password is incorrect')
