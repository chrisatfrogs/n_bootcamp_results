import json
import pandas as pd
import streamlit as st
from turn import TurnDataset
from constants import TURNS



# Set the page layout to wide
st.set_page_config(layout="wide", page_title="Turn results")

with st.sidebar:
    st.markdown('# :tada: Turn result dashboard :tada:')
    turn = st.selectbox('Select a turn', TURNS)
    st.markdown('<br/><br/>', unsafe_allow_html=True)




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


def main():
    
    data = load_data(turn)

    # Load attributes
    turn_type = data[turn]['type']
    dataset = data[turn]['dataset']
    models = data[turn]['models']
    research_objectives = data[turn]['research_objectives']
    research_findings = data[turn]['research_findings']
    
    # Load dataset
    turn_dataset = TurnDataset(dataset, turn, models, turn_type)

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
    st.markdown('## Research findings')
    st.markdown(research_findings, unsafe_allow_html=True)

    # Type-dependent graphs
    if turn_type == 'non-fiction':

        # Mean summary section
        mean_summary = turn_dataset.eval_frame().T
        mean_summary = mean_summary.rename(columns=models)
        st.markdown('## Mean summary')
        column_selection = st.multiselect('Select models', mean_summary.columns)
        if column_selection:
            st.dataframe(mean_summary[column_selection].style.highlight_max(axis=1, color='#6493CC').format(precision=2), use_container_width=True)
        else:
            st.dataframe(mean_summary.style.highlight_max(axis=1, color='#6493CC').format(precision=2), use_container_width=True)

        st.markdown('<br/>', unsafe_allow_html=True)

        # Bar chart section
        st.markdown('## Bar chart')
        bar_chart = turn_dataset.plotly_bar_chart()
        st.plotly_chart(bar_chart, use_container_width=True)
        st.markdown('<br/>', unsafe_allow_html=True)

        # Box plots section
        st.markdown('## Box plots')
        with st.expander('See charts'):
            st.plotly_chart(turn_dataset.plotly_box_plot(), use_container_width=True)
        
        st.markdown('<br/>', unsafe_allow_html=True)

        # Significance table section
        st.markdown('## Significance table')
        significance_table = turn_dataset.generate_significance_table()
        st.markdown(significance_table, unsafe_allow_html=True)
        st.markdown('<br/>', unsafe_allow_html=True)

        # Fact check section
        st.markdown('## Explicit and implicit fact checks')
        col10, col11 = st.columns(2)
        with col10:
            st.plotly_chart(turn_dataset.plotly_pie_chart('factual_correctness'), use_container_width=True)
        
        with col11:
            st.plotly_chart(turn_dataset.plotly_pie_chart('implicit_fact_check'), use_container_width=True)


    elif turn_type == 'fiction':
        pass

    elif turn_type == 'horoscope':
        # Mean summary section
        mean_summary = turn_dataset.eval_frame().T
        mean_summary = mean_summary.rename(columns=models)
        st.markdown('## Mean summary')
        column_selection = st.multiselect('Select models', mean_summary.columns)
        if column_selection:
            st.dataframe(mean_summary[column_selection].style.highlight_max(axis=1, color='#6493CC').format(precision=2), use_container_width=True)
        else:
            st.dataframe(mean_summary.style.highlight_max(axis=1, color='#6493CC').format(precision=2), use_container_width=True)

        st.markdown('<br/>', unsafe_allow_html=True)

        # Bar chart section
        st.markdown('## Bar chart')
        bar_chart = turn_dataset.plotly_bar_chart()
        st.plotly_chart(bar_chart, use_container_width=True)
        st.markdown('<br/>', unsafe_allow_html=True)

        # Significance table section
        st.markdown('## Significance table')
        significance_table = turn_dataset.generate_significance_table()
        st.markdown(significance_table, unsafe_allow_html=True)
        st.markdown('<br/>', unsafe_allow_html=True)

        # Box plots section
        st.markdown('## Box plots')
        st.plotly_chart(turn_dataset.plotly_box_plot(), use_container_width=True)
        st.markdown('<br/>', unsafe_allow_html=True)


    



if __name__ == '__main__':
    main()