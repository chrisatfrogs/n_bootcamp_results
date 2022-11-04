import re
import math
from itertools import combinations

import pandas as pd
import numpy as np
import seaborn as sns
import plotly.graph_objects as go

from scipy.stats import t
from plotly.subplots import make_subplots

from constants import RATING_CRITERIA_COLS, AVG_RATING_COL, COLS, GUEST_USER_ID, EXCLUDE_USER_IDS

class TurnDataset:
    def __init__(self, 
        turn: str, 
        df: pd.DataFrame, 
        models: dict, 
        turn_type: str):
        
        
        self.turn = turn
        self.df = df
        self.turn_type = turn_type
        self.models = models
        self.clean_df()
        self.rating_criterias = self.get_rating_criterias()
        
    def clean_df(self) -> pd.DataFrame:
        self.df['turn'] = self.turn
        self.df = self.df[~self.df[COLS['user']].isin(EXCLUDE_USER_IDS)]
        self.df = self.df.drop(columns=['context_realization', 'matches_fandom'])
        self.df = self.df.replace(-1, np.nan)
        if 'model' not in self.df.columns:
            assert COLS['listvalue'] in self.df.columns, 'No listvalue column in the dataframe.'
            self.df['model'] = self.df[COLS['listvalue']].apply(lambda x: self.models[x])

    def get_general_info(self) -> tuple:
        n_texts = self.df[COLS['story_id']].nunique()
        n_raters = self.df[COLS['user']].nunique()
        n_ratings = self.df.shape[0]

        return n_texts, n_raters, n_ratings
    
    def get_rating_criterias(self):
        cols_with_nans = set(self.df.columns[self.df.isnull().all()])
        possible_rating_cols = set(RATING_CRITERIA_COLS + [AVG_RATING_COL])
        rating_criterias = list(possible_rating_cols.intersection(self.df.columns) - cols_with_nans)
        return rating_criterias
    
    def get_data(self, 
        unbiased: bool = True, 
        user_independent: bool = False) -> pd.DataFrame:
        
        data = self.df.copy()

        if unbiased:
            mean_ratings = (
                data[data[COLS['user']] != GUEST_USER_ID].groupby(COLS['user'])[self.rating_criterias].mean()
            )

            for criterion in self.rating_criterias:
                meanbias = mean_ratings[criterion].mean()
                unbiases = meanbias / mean_ratings[criterion]
                data[criterion] = data.apply(lambda row:
                    row[criterion] if row[COLS['user']] == GUEST_USER_ID
                    else row[criterion] * unbiases[row[COLS['user']]], axis=1)
        
        if user_independent:
            data = data.groupby([COLS['story_id']], dropna=False).mean()
            data = data.reset_index()
        
        return data

    def _get_plot_df(self) -> pd.DataFrame:
        groupby_cols = [COLS['turn'], 'model']

        data = self.get_data()

        data = self.df.groupby([COLS['story_id']] + groupby_cols, dropna=False, as_index=False).mean()
        data = pd.melt(
            data,
            id_vars=[COLS['story_id']] + groupby_cols,
            value_vars=self.rating_criterias
        )

        data = data.groupby(['model']).apply(lambda x: x.sort_values(['variable'], ascending=True)).reset_index(drop=True)

        return data 
    
    def eval_frame(self):
        """
        Evaluate data in a frame format.
        """
        groupby_cols = ['model']

        df = self.get_data(unbiased=True)
        df = df.groupby(groupby_cols, dropna=False)[sorted(self.rating_criterias)].mean()
        return df
    
    def plotly_bar_chart(self):
        """
        Plot data in a bar chart.
        """
        df = self._get_plot_df()
        df = df.groupby(['model', 'variable'], dropna=False)['value'].mean().reset_index()
        fig = go.Figure()
        for group in df['model'].unique():
            fig.add_trace(go.Bar(
                x=df[df['model'] == group]['variable'],
                y=df[df['model'] == group]['value'],
                name=group
            ))
        
        fig.update_layout(barmode='group', boxmode='group', yaxis={'range': [0, 100]})
        return fig
    
    
    def plotly_box_plot(self):
        """
        Plot data in a box plot.
        """
        
        if self.turn_type == 'non-fiction':
            nf_aux_criteria = ['action_coherence',
                            'citation_correct',
                            'exhaustive_information',
                            'factual_correctness',
                            'grammar',
                            'identical_information',
                            'implicit_fact_check',
                            'readability'
                            ]
            nf_main_criteria = [
                                'content_similarity',
                                'linguistic_difference',
                                'overall_quality'
                            ]
            
            fig = make_subplots(
                rows=7, cols=2,
                specs = [
                    [{}, {}],
                    [{}, {}],
                    [{}, {}],
                    [{}, {}],
                    [{"colspan": 2}, None],
                    [{"colspan": 2}, None],
                    [{"colspan": 2}, None],
                ],
                subplot_titles=nf_aux_criteria + nf_main_criteria
            )

            df = self.get_data(unbiased=True)
            has_legend = False
            for i, criterion in (enumerate(nf_aux_criteria + nf_main_criteria)):
                if criterion in nf_aux_criteria:
                    temp_df = df[[criterion, 'model']]
                    group_labels = sorted(temp_df['model'].unique().tolist())
                    group_data = [temp_df[temp_df['model'] == label][criterion] for label in group_labels]
                    if i % 2 == 0:
                        row = i // 2 + 1
                        col = 1           
                    else:
                        row = i // 2 + 1
                        col = 2
                    for label, data in zip(group_labels, group_data):
                        if not has_legend:
                            fig.add_trace(go.Box(y=data, name=label, boxmean='sd', showlegend=True, jitter = 0.5, legendgroup=label), row=row, col=col)
                        else:
                            fig.add_trace(go.Box(y=data, name=label, boxmean='sd', showlegend=False, jitter = 0.5, legendgroup=label), row=row, col=col)
        
                else:
                    temp_df = df[[criterion, 'model']]
                    group_labels = sorted(temp_df['model'].unique().tolist())
                    group_data = [temp_df[temp_df['model'] == label][criterion] for label in group_labels]
                    for label, data in zip(group_labels, group_data):
                        fig.add_trace(go.Box(x=data, name=label, boxmean='sd', jitter=0.5, showlegend=False, legendgroup=label), row=i-3, col=1)
                has_legend = True
            # names = set()
            # fig.for_each_trace(lambda trace: trace.update(showlegend=False) if (trace.name in names) else names.add(trace.name))
            fig.update_layout(height=1500, width=1000, title_text="Distribution of rating criteria")

        elif self.turn_type == 'horoscope':
            fig = make_subplots(
                rows=4, cols=2,
                specs = [
                    [{}, {}],
                    [{}, {}],
                    [{}, {}],
                    [{"colspan": 2}, None],
                ],
                subplot_titles=self.rating_criterias
            )
            df = self.get_data(unbiased=True)
            has_legend = False
            for i, criterion in enumerate(self.rating_criterias):
                temp_df = df[[criterion, 'model']]
                group_labels = sorted(temp_df['model'].unique().tolist())
                group_data = [temp_df[temp_df['model'] == label][criterion] for label in group_labels]
                if i % 2 == 0:
                    row = i // 2 + 1
                    col = 1           
                else:
                    row = i // 2 + 1
                    col = 2
                for label, data in zip(group_labels, group_data):
                    if not has_legend:
                        fig.add_trace(go.Box(y=data, name=label, boxmean='sd', showlegend=True, jitter = 0.5, legendgroup=label), row=row, col=col)
                    else:
                        fig.add_trace(go.Box(y=data, name=label, boxmean='sd', showlegend=False, jitter = 0.5, legendgroup=label), row=row, col=col)
                has_legend = True
            

        fig.update_layout(height=1500, width=1000, title_text="Distribution of rating criteria")
        return fig
    
    @staticmethod
    def generate_plotly_table(model_mapper: dict, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate a table with the data.
        """
        data = {'List value': [int(lv) for lv in model_mapper.keys()], 
            'Model': [model_mapper[lv] for lv in model_mapper.keys()]}
        result_df = pd.DataFrame(data, index = None)
        assert 'listvalue' in df.columns, 'listvalue not in dataframe'
        assert 'story_id' in df.columns, 'story_id not in dataframe'
        result_df['Number of ratings'] = [len(df[df['listvalue'] == lv]) for lv in model_mapper.keys() ]
        result_df['Number of stories'] = [df[df['listvalue'] == lv]['story_id'].nunique() for lv in model_mapper.keys()]

        return result_df
    
    def significance_table(self):
        df = self.get_data(unbiased=True)
        criterias = sorted(self.rating_criterias)
        models = sorted(df['model'].unique().tolist())
        significance_table = pd.DataFrame(index=criterias)
        model_combinations = list(combinations(models, 2))
        style_dict = {}

        for model_combination in model_combinations:
            data1 = df[df['model'] == model_combination[0]]
            data2 = df[df['model'] == model_combination[1]]
            degree_of_freedom = len(data1) + len(data2) - 2
            col_name = f'{model_combination[0]} vs {model_combination[1]}'

            for criterion in criterias:
                mean1, mean2 = data1[criterion].mean(), data2[criterion].mean()
                std1, std2 = data1[criterion].std(), data2[criterion].std()
                se1, se2 = std1 / np.sqrt(len(data1)), std2 / np.sqrt(len(data2))
                se = np.sqrt(se1**2 + se2**2)
                t_stat = (mean1 - mean2) / se
                critical_value = t.ppf(1.0 - 0.05, degree_of_freedom)
                p_value = (1.0 - t.cdf(abs(t_stat), degree_of_freedom)) * 2.0

                significance_table.loc[criterion, col_name] = f't_statistic: {t_stat:.4f}<br> p-value: {p_value:.4f}'
                style_dict[(criterion, col_name)] = (critical_value, 0.05)
        
        return significance_table, style_dict
    
    @staticmethod
    def style_significance_table(df: pd.DataFrame, style_dict: dict):
        
        """
        Style the significance table.
        """
        styled_df = pd.DataFrame('', index = df.index, columns = df.columns)
        for (criteria, col_name), (critical_value, p_value) in style_dict.items():
            t_statistic, p_value = re.compile(r't_statistic: (.*)<br> p-value: (.*)').search(df.at[criteria, col_name]).groups()
            if abs(float(t_statistic)) >= critical_value or float(p_value) <= 0.05:
                styled_df.at[criteria, col_name] = 'background-color: #ccff99;'
            else:
                styled_df.at[criteria, col_name] = 'opacity: 0.2;'
            if abs(float(t_statistic)) >= critical_value and float(p_value) <= 0.05:
                # This indicates that the values denote a significant difference between the list values. 
                styled_df.at[criteria, col_name] = 'background-color: #2f5e00; color: #ffffff'

        return styled_df
    
    def generate_significance_table(self):
        df, style_dict = self.significance_table()
        significance_table = df.style.apply(self.style_significance_table, style_dict=style_dict, axis=None)
        significance_table.set_properties(**{'text-align': 'center', 'font-family': 'Calibri','font-size': '10px'})
        significance_table = significance_table.to_html()
        significance_table = significance_table.replace('font-family: Calibri;', "font-family: 'Source Sans Pro', sans-serif;")
        return significance_table
    
    def plotly_pie_chart(self, column: str):
        df = self.df.copy()
        fact_check_cols = ['factual_correctness', 'implicit_fact_check']
        error_dict = {100: '0 error (100%)', 
            75: '1 error (75%)', 
            50: '2 errors (50%)', 
            25: '3 errors (25%)',
            0: '> 3 errors (0%)'}
        df = df[[COLS['listvalue']] + fact_check_cols]
        df = df.replace({'factual_correctness': error_dict, 'implicit_fact_check': error_dict})
        for col in fact_check_cols:
            df = df[df[col].isin(error_dict.values())]
        lvs = sorted(df['listvalue'].unique().tolist())
        rows =  math.ceil(len(lvs) / 2)
        cols = 2
        fig = make_subplots(rows=rows, cols=cols, specs=[[{'type':'domain'}, {'type':'domain'}]]*rows)
        for i, lv in enumerate(lvs):
            data = df[df['listvalue'] == lv]
            labels = data[column].value_counts().index.tolist()
            values = data[column].value_counts().values.tolist()
            row = i // 2 + 1
            col = i % 2 + 1
            fig.add_trace(go.Pie(labels=labels, values=values, name=f'List value {lv}'), row=row, col=col)
        
        fig.update_traces(hole=.4, hoverinfo="label+percent+name")
        fig.update_layout(
            title_text=f'Pie chart of {column}',
        )
        return fig



    @staticmethod
    def unbias_1(df: pd.DataFrame, criteria: list) -> pd.DataFrame:
        
        assert 'user' in df.columns, "user not in dataframe"
        assert 'story_id' in df.columns, "story_id not in dataframe"
        assert len(set(df.columns).intersection(set(criteria))) == len(criteria), "not all criteria columns in dataframe"
        
        bias_dict = {}
        for criterion in criteria:
            temp_df = df.groupby([COLS['story_id']])[criterion].mean()
            bias_dict[criterion] = temp_df.to_dict()


        user_net_bias = {}
        users = df.user.unique().tolist()
        for user in users:
            temp_df = df[df.user == user]
            criteria_dict = {}
            for criterion in criteria:
                sub_temp_df = temp_df[[COLS['story_id'], criterion]]
                sub_temp_df['bias'] = sub_temp_df.apply(lambda row: bias_dict[criterion][row[COLS['story_id']]], axis = 1)
                sub_temp_df['bias_rating'] = sub_temp_df.apply(lambda row: row[criterion] - row['bias'], axis = 1)
                criteria_dict[criterion] = sub_temp_df['bias_rating'].mean()
            
            user_net_bias[user] = criteria_dict
        
        adjusted_df = df.copy()
        for user in users:
            user_criteria_dict = user_net_bias[user]
            for criterion in criteria:
                adjusted_df.loc[adjusted_df.user == user, criterion] = adjusted_df.loc[adjusted_df.user == user, criterion] - user_criteria_dict[criterion]
        
        return adjusted_df
    
    @staticmethod
    def unbias_2(df: pd.DataFrame, criteria: list) -> pd.DataFrame:
        assert 'user' in df.columns, "user not in dataframe"
        assert 'story_id' in df.columns, "story_id not in dataframe"
        assert len(set(df.columns).intersection(set(criteria))) == len(criteria), "not all criteria columns in dataframe"

        def _get_mean_dict(col: str, criteria: list) -> pd.DataFrame:
            unique_cols = df[col].unique().tolist()
            mean_dict = {}
            for unique_col in unique_cols:
                temp_df = df[df[col] == unique_col]
                mean_dict[unique_col] = temp_df[criteria].mean().to_dict()
            return mean_dict
        
        def _get_bias_dict(col: str, criteria: list, mean_dict: dict) -> pd.DataFrame:
            unique_cols = df[col].unique().tolist()
            bias_dict = {}
            for unique_col in unique_cols:
                temp_df = df[df[col] == unique_col]
                bias_dict[unique_col] = {}
                for criterion in criteria:
                    temp_df[criterion + '_bias'] = temp_df[criterion] - mean_dict[unique_col][criterion]
                    bias_dict[unique_col][criterion] = temp_df[criterion + '_bias'].mean()
            return bias_dict
        
        user_mean_dict = _get_mean_dict(COLS['user'], criteria)
        story_mean_dict = _get_mean_dict(COLS['story_id'], criteria)
        user_bias_dict = _get_bias_dict(COLS['user'], criteria, user_mean_dict)
        story_bias_dict = _get_bias_dict(COLS['story_id'], criteria, story_mean_dict)

        adjusted_df = df.copy()
        for ix, row in adjusted_df.iterrows():
            for criterion in criteria:
                user_tendency = user_bias_dict[row[COLS['user']]][criterion]
                story_tendency = story_bias_dict[row[COLS['story_id']]][criterion]
                user_mean_criterion = user_mean_dict[row[COLS['user']]][criterion]
                story_mean_criterion = story_mean_dict[row[COLS['story_id']]][criterion]
                if user_tendency > 0 and story_tendency > 0:
                    adjusted_df.loc[ix, criterion] = max(user_mean_criterion + story_tendency, story_mean_criterion + user_tendency)
                elif user_tendency < 0 and story_tendency < 0:
                    adjusted_df.loc[ix, criterion] = min(user_mean_criterion + story_tendency, story_mean_criterion + user_tendency)
                elif user_tendency < 0 and story_tendency > 0:
                    if user_mean_criterion <= story_mean_criterion:
                        adjusted_value = (story_mean_criterion + user_tendency) * 0.5 + (user_mean_criterion + story_tendency) * 0.5
                        adjusted_df.loc[ix, criterion] = min(max(user_mean_criterion, adjusted_value), story_mean_criterion)
                    else:
                        adjusted_value = story_mean_criterion * 0.5 + user_mean_criterion * 0.5
                        adjusted_df.loc[ix, criterion] = adjusted_value
                elif user_tendency > 0 and story_tendency < 0:
                    if user_mean_criterion > story_mean_criterion:
                        adjusted_value = (story_mean_criterion + user_tendency) * 0.5 + (user_mean_criterion + story_tendency) * 0.5
                        adjusted_df.loc[ix, criterion] = max(min(story_mean_criterion, adjusted_value), user_mean_criterion)
                    else:
                        adjusted_value = story_mean_criterion * 0.5 + user_mean_criterion * 0.5
                        adjusted_df.loc[ix, criterion] = adjusted_value

        return adjusted_df
    