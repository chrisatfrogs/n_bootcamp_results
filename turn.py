import re
import math
from itertools import combinations

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from scipy.stats import t
from plotly.subplots import make_subplots

from constants import RATING_CRITERIA_COLS, AVG_RATING_COL, COLS, GUEST_USER_ID, EXCLUDE_USER_IDS

class Dataset:
    '''
    Base class for datasets.
    '''

    def __init__(self,
        turn: str,
        data_source: str,
        models: dict,
        text_source: str = None
        ):
        
        self.turn = turn
        self.data_source = data_source
        self.models = models
        self.text_source = text_source if text_source else None
        self.rating_criteria = []
        self.data = pd.read_csv(self.data_source)
        self.text_df = pd.read_csv(self.text_source) if self.text_source else None
    
    def get_general_info(self) -> tuple:
        '''
        Returns general information about the dataset.

        Returns:
        --------
        general_info: tuple
            Tuple with general information about the dataset containing the following:
            - number of texts
            - number of users
            - number of ratings

        '''
        n_texts = self.data[COLS['story_id']].nunique()
        n_raters = self.data[COLS['user']].nunique()
        n_ratings = self.data.shape[0]

        return n_texts, n_raters, n_ratings
    
    def get_list_info(self) -> pd.DataFrame:
        '''
        Returns a dataframe with list value-specific information

        Returns:
        --------
        list_info: pd.DataFrame
            Dataframe with list value-specific information. Contains the following columns:
            - List value
            - Model
            - Number of ratings
            - Number of stories
        '''
        assert 'listvalue' in self.data.columns, 'listvalue not in dataframe'
        assert 'story_id' in self.data.columns, 'story_id not in dataframe'

        list_info_df = pd.DataFrame(columns = ['List value', 'Model', 'Number of ratings', 'Number of stories'])
        list_info_df['List value'] = [int(lv) for lv in self.models.keys()]
        list_info_df['Model'] = [self.models[lv] for lv in self.models.keys()]
        list_info_df['Number of ratings'] = [self.data[self.data[COLS['model']] == model].shape[0] for model in self.models.values()]
        list_info_df['Number of stories'] = [self.data[self.data[COLS['model']] == model]['story_id'].nunique() for model in self.models.values()]

        return list_info_df

    def get_unbiased_data(self, unbias_method: str = 'overall_mean') -> pd.DataFrame:
        '''
        Returns a dataframe with unbiased data.

        Parameters:
        -----------
        unbias_method: str (default: 'overall_mean')
            Method to use for unbiased data. Options are:
            - 'overall_mean'
            - 'user_mean'
            - 'item_mean'
        
        Returns:
        --------
        unbiased_df: pd.DataFrame
            Dataframe with unbiased ratings.

        '''
        unbiased_df = self.data.copy()
        if unbias_method == 'overall_mean':
            unbiased_df = self.unbias_by_overall_mean(self.data, self.rating_criteria)
        
        elif unbias_method == 'story_mean':
            unbiased_df = self.unbias_by_story_mean(self.data, self.rating_criteria)
        
        elif unbias_method == 'user_story_mean':
            unbiased_df = self.unbias_by_user_story_mean(self.data, self.rating_criteria)
        
        return unbiased_df
    
    @staticmethod
    def unbias_by_overall_mean(df: pd.DataFrame, criteria: list) -> pd.DataFrame:
        '''
        Returns unbiased ratings in a dataframe

        Gets the unbiased ratings by computing the mean bias per criterion per user and multiplying the mean bias by the user's actual rating.

        Parameters:
        -----------
        df: pd.DataFrame
            Dataframe with ratings.
        criteria: list
            List of criteria to use for unbiased ratings.
        
        Returns:
        --------
        unbiased_df: pd.DataFrame
            Dataframe with unbiased ratings.

        '''
        assert COLS['user'] in df.columns, 'User column not found in dataframe.'
        adjusted_df = df.copy()
        mean_ratings = adjusted_df.groupby(COLS['user'])[criteria].mean()
        
        for criterion in criteria:
            meanbias = mean_ratings[criterion].mean()
            unbiases = meanbias / mean_ratings[criterion]
            adjusted_df[criterion] = adjusted_df.apply(lambda row: row[criterion] * unbiases[row[COLS['user']]], axis=1)

        return adjusted_df
    
    @staticmethod
    def unbias_by_story_mean(df: pd.DataFrame, criteria: list) -> pd.DataFrame:
        '''
        Returns unbiased ratings in a dataframe

        The core assumption here is that the average rating for each story is unbiased. The unbiased rating for each user is then calculated
        based on this core assumption where the user's rating is subtracted from the average rating for each story. These differences are then
        averaged over all stories which gives the net bias rating for each user. This net bias rating is then subtracted from the user's actual
        rating.

        References:
        Wadbude, R., Gupta, V., Mekala, D., & Karnick, H. (2018). User bias removal in review score prediction. In Proceedings of the ACM India 
        Joint International Conference on Data Science and Management of Data. CoDS-COMAD ’18: The ACM India Joint International Conference on 
        Data Science & Management of Data. ACM. https://doi.org/10.1145/3152494.3152520

        Parameters:
        -----------
        df: pd.DataFrame
            Dataframe with ratings.
        criteria: list
            List of criteria to use for unbiased ratings.
        
        Returns:
        --------
        unbiased_df: pd.DataFrame
            Dataframe with unbiased ratings.
        '''
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
    def unbias_by_user_story_mean(df: pd.DataFrame, criteria: list) -> pd.DataFrame:
        '''
        Returns unbiased ratings in a dataframe

        This unbias method is an extension of the story mean unbias method. The core assumption here is that each user has a certain tendency to rate stories
        positively or negatively. Similarly, each story is assumed to be considered by the underlying population of users as particularly good or bad. The
        user and story tendencies are then used to unbias the ratings.

        References:
        Pranshi Yadav, Priya Yadav, Pegah Nokhiz, and Vivek Gupta. 2020. Unbiasing Review Ratings with Tendency Based Collaborative Filtering. In Proceedings of
        the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural
        Language Processing: Student Research Workshop, pages 50–56, Suzhou, China. Association for Computational Linguistics.

        Parameters:
        -----------
        df: pd.DataFrame
            Dataframe with ratings.
        criteria: list
            List of criteria to use for unbiased ratings.
        
        Returns:
        --------
        unbiased_df: pd.DataFrame
            Dataframe with unbiased ratings.

        '''
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
    
    def get_mean_table(self, unbias_method: str = 'overall_mean') -> pd.DataFrame:
        '''
        Returns a dataframe with mean ratings for each criterion.

        Parameters:
        -----------
        unbias_method: str
            Method to use for unbiasing ratings. Options are:
            - overall_mean
            - story_mean
            - user_story_mean
        
        Returns:
        --------
        mean_frame: pd.DataFrame
            Dataframe with mean ratings for each criterion.
        '''
        groupby_cols = [COLS['model']]

        mean_df = self.get_unbiased_data(unbias_method)
        mean_df = mean_df.groupby(groupby_cols, dropna=False)[self.rating_criteria].mean().T
        return mean_df
    
    def get_bar_chart(self):
        '''
        Returns a bar chart with mean ratings for each criterion.
        
        Returns:
        --------
        fig: go._figure.Figure
            Bar chart with mean ratings for each criterion.
        '''
        models = self.data[COLS['model']].unique().tolist()
        df = self.get_unbiased_data()
        fig = go.Figure()
        for model in models:
            model_df = df[df[COLS['model']] == model]
            fig.add_trace(go.Bar(x=self.rating_criteria, y=model_df[self.rating_criteria].mean().values, name=model))
        
        fig.update_layout(barmode='group', boxmode='group', yaxis={'range': [0, 100]})
        return fig
        
    def generate_significance_table(self):
        '''
        Generates a table containing significance values for each criterion, together with a style dictionary.

        Returns:
        --------
        significance_table: pd.DataFrame
            Dataframe with significance values for each criterion.
        
        style_dict: dict
            Dictionary with style information for the significance table.
        '''

        df = self.get_unbiased_data()
        criteria = sorted(self.rating_criteria)
        models = sorted(df[COLS['model']].unique().tolist())
        significance_table = pd.DataFrame(index=criteria)
        model_combinations = list(combinations(models, 2))
        style_dict = {}

        for model_combination in model_combinations:
            data1 = df[df['model'] == model_combination[0]]
            data2 = df[df['model'] == model_combination[1]]
            degree_of_freedom = len(data1) + len(data2) - 2
            col_name = f'{model_combination[0]} vs {model_combination[1]}'

            for criterion in self.rating_criteria:
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
    def style_significance_table(df: pd.DataFrame, style_dict: dict) -> pd.DataFrame:
        '''
        Styles a significance table.

        Parameters:
        -----------
        df: pd.DataFrame
            Dataframe with significance values for each criterion.
        
        style_dict: dict
            Dictionary with style information for the significance table.
        
        Returns:
        --------
        df: pd.DataFrame
            Styled dataframe with significance values for each criterion.
        '''
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

    def get_significance_table(self) -> str:
        '''
        Returns an HTML table containing significance values for each criterion.

        Returns:
        --------
        significance_table: str
            Dataframe with significance values for each criterion.
        '''
        significance_table, style_dict = self.generate_significance_table()
        significance_table = significance_table.style.apply(self.style_significance_table, style_dict=style_dict, axis=None)
        significance_table.set_properties(**{'text-align': 'center', 'font-family': 'Calibri','font-size': '10px'})
        significance_table = significance_table.to_html()
        significance_table = significance_table.replace('font-family: Calibri;', "font-family: 'Source Sans Pro', sans-serif;")
        return significance_table



class OldNonFictionDataset(Dataset):
    def __init__(self,
        turn: str,
        data_source: str,
        models: dict,
        text_source: str = None
        ):

        super().__init__(turn, data_source, models, text_source)
        self.aux_criteria = [
                            'action_coherence',
                            'citation_correct',
                            'exhaustive_information',
                            'factual_correctness',
                            'grammar',
                            'identical_information',
                            'implicit_fact_check',
                            'readability'
                            ]
        self.main_criteria = [
                            'content_similarity',
                            'linguistic_difference',
                            'overall_quality'
                            ]
        self.rating_criteria = self.aux_criteria + self.main_criteria
        self.clean_df()
    
    def clean_df(self):
        '''
        Cleans the dataframe by removing rows with missing values and rows with a rating of 0.

        Parameters:
        -----------
        df: pd.DataFrame
            Dataframe to be cleaned.
        
        Returns:
        --------
        df: pd.DataFrame
            Cleaned dataframe.
        '''
        self.data['turn'] = self.turn
        self.data = self.data[~self.data[COLS['user']].isin(EXCLUDE_USER_IDS)]
        self.data = self.data.drop(columns=['context_realization', 'matches_fandom'])
        self.data = self.data.replace(-1, np.nan)
        if COLS['model'] not in self.data.columns:
            assert COLS['listvalue'] in self.data.columns, 'No listvalue column in the dataframe.'
            self.data[COLS['model']] = self.data[COLS['listvalue']].apply(lambda x: self.models[x])
    
    def get_box_plot(self):
        '''
        Returns a box plot of the ratings for each criterion.

        Returns:
        --------
        box_plot: go._figure.Figure
            Box plot of the ratings for each criterion.

        '''
        df = self.get_unbiased_data()
        double_rows = math.ceil(len(self.aux_criteria) / 2) 
        single_rows = len(self.main_criteria)
        total_rows = double_rows + single_rows
        specs = [[{}, {}]] * double_rows + [[{'colspan': 2}, None]] * single_rows
        fig = make_subplots(rows=total_rows, cols=2, specs=specs, subplot_titles=self.rating_criteria)
        has_legend = False
        for i, criterion in enumerate(self.rating_criteria):
            criterion_df = df[[criterion, COLS['model']]]
            group_labels = sorted(criterion_df[COLS['model']].unique().tolist())
            group_data = [criterion_df[criterion_df[COLS['model']] == group_label][criterion].tolist() for group_label in group_labels]
            if criterion in self.aux_criteria:
                row = i // 2 + 1
                col = 1 if i % 2 == 0 else 2
            else:
                row = i - 3
                col = 1
                
            for label, data in zip(group_labels, group_data):
                if not has_legend:
                    fig.add_trace(go.Box(y=data, name=label, boxmean='sd', showlegend=True, jitter=0.5, legendgroup=label), row=row, col=col)
                else:
                    fig.add_trace(go.Box(y=data, name=label, boxmean='sd', showlegend=False, jitter=0.5, legendgroup=label), row=row, col=col)

            has_legend = True
        
        fig.update_layout(height=total_rows * 200, width=1000, title_text="Distribution of rating criteria")
        return fig

    def get_pie_chart(self, column: str):
        '''
        Returns a pie chart of the distribution of the values in a column.

        Parameters:
        -----------
        column: str
            Column name of the column for which the pie chart should be generated.
            Can only be of the following values:
            - 'factual_correctness'
            - 'implicit_fact_check'
        
        Returns:
        --------
        pie_chart: go._figure.Figure
            Pie chart of the distribution of the values in a column.
        
        '''

        df = self.data.copy()
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


class NewNonFictionDataset(Dataset):
    def __init__(self,
        turn: str,
        data_source: str,
        models: dict,
        text_source: str = None
        ):

        super().__init__(turn, data_source, models, text_source)
        self.aux_criteria = [
                            'action_coherence',
                            'citation_correct',
                            'exhaustive_information',
                            'factual_correctness',
                            'grammar',
                            'identical_information',
                            'lexical_correctness',
                            'syntactic_correctness',
                            ]
        self.main_criteria = [
                            'content_similarity',
                            'linguistic_difference',
                            'overall_quality'
                            ]
        self.rating_criteria = self.aux_criteria + self.main_criteria
        self.clean_df()
    
    def clean_df(self):
        '''
        Cleans the dataframe by removing rows with missing values and rows with a rating of 0.

        Parameters:
        -----------
        df: pd.DataFrame
            Dataframe to be cleaned.
        
        Returns:
        --------
        df: pd.DataFrame
            Cleaned dataframe.
        '''
        self.data['turn'] = self.turn
        self.data = self.data[~self.data[COLS['user']].isin(EXCLUDE_USER_IDS)]
        self.data = self.data.drop(columns=['context_realization', 'matches_fandom'])
        self.data = self.data.replace(-1, np.nan)
        if COLS['model'] not in self.data.columns:
            assert COLS['listvalue'] in self.data.columns, 'No listvalue column in the dataframe.'
            self.data[COLS['model']] = self.data[COLS['listvalue']].apply(lambda x: self.models[x])

    def get_box_plot(self):
        '''
        Returns a box plot of the ratings for each criterion.

        Returns:
        --------
        box_plot: go._figure.Figure
            Box plot of the ratings for each criterion.

        '''
        df = self.get_unbiased_data()
        double_rows = math.ceil(len(self.aux_criteria) / 2) 
        single_rows = len(self.main_criteria)
        total_rows = double_rows + single_rows
        specs = [[{}, {}]] * double_rows + [[{'colspan': 2}, None]] * single_rows
        fig = make_subplots(rows=total_rows, cols=2, specs=specs, subplot_titles=self.rating_criteria)
        has_legend = False
        for i, criterion in enumerate(self.rating_criteria):
            criterion_df = df[[criterion, COLS['model']]]
            group_labels = sorted(criterion_df[COLS['model']].unique().tolist())
            group_data = [criterion_df[criterion_df[COLS['model']] == group_label][criterion].tolist() for group_label in group_labels]
            if criterion in self.aux_criteria:
                row = i // 2 + 1
                col = 1 if i % 2 == 0 else 2
            else:
                row = i - 3
                col = 1
                
            for label, data in zip(group_labels, group_data):
                if not has_legend:
                    fig.add_trace(go.Box(y=data, name=label, boxmean='sd', showlegend=True, jitter=0.5, legendgroup=label), row=row, col=col)
                else:
                    fig.add_trace(go.Box(y=data, name=label, boxmean='sd', showlegend=False, jitter=0.5, legendgroup=label), row=row, col=col)

            has_legend = True
        
        fig.update_layout(height=total_rows * 200, width=1000, title_text="Distribution of rating criteria")
        return fig
    
    
class HoroscopeDataset(Dataset):
    def __init__(self,
        turn: str,
        data_source: str,
        models: dict,
        text_source: str = None
        ):

        super().__init__(turn, data_source, models, text_source)
        self.rating_criteria = [
                                'info_consistency',
                                'matches_style_tone',
                                'readability',
                                'spelling_and_grammar',
                                'story_coherence',
                                'topic_met'
                                ]
        self.clean_df()

    def clean_df(self):
        '''
        Cleans the dataframe by removing rows with missing values and rows with a rating of 0.

        Parameters:
        -----------
        df: pd.DataFrame
            Dataframe to be cleaned.
        
        Returns:
        --------
        df: pd.DataFrame
            Cleaned dataframe.
        '''
        self.data['turn'] = self.turn
        self.data = self.data[~self.data[COLS['user']].isin(EXCLUDE_USER_IDS)]
        self.data = self.data.drop(columns=['context_realization', 'matches_fandom'])
        self.data = self.data.replace(-1, np.nan)
        if COLS['model'] not in self.data.columns:
            assert COLS['listvalue'] in self.data.columns, 'No listvalue column in the dataframe.'
            self.data[COLS['model']] = self.data[COLS['listvalue']].apply(lambda x: self.models[x])

    def get_box_plot(self):
        '''
        Returns a box plot of the ratings for each criterion.

        Returns:
        --------
        box_plot: go._figure.Figure
            Box plot of the ratings for each criterion.

        '''
        df = self.get_unbiased_data()
        total_rows = math.ceil(len(self.rating_criteria) / 2)
        specs = [[{}, {}]] * total_rows
        fig = make_subplots(rows=total_rows, cols=2, specs=specs, subplot_titles=self.rating_criteria)
        has_legend = False
        for i, criterion in enumerate(self.rating_criteria):
            criterion_df = df[[criterion, COLS['model']]]
            group_labels = sorted(criterion_df[COLS['model']].unique().tolist())
            group_data = [criterion_df[criterion_df[COLS['model']] == group_label][criterion].tolist() for group_label in group_labels]
            row = i // 2 + 1
            col = i % 2 + 1
                
            for label, data in zip(group_labels, group_data):
                if not has_legend:
                    fig.add_trace(go.Box(y=data, name=label, boxmean='sd', showlegend=True, jitter=0.5, legendgroup=label), row=row, col=col)
                else:
                    fig.add_trace(go.Box(y=data, name=label, boxmean='sd', showlegend=False, jitter=0.5, legendgroup=label), row=row, col=col)

            has_legend = True
        
        fig.update_layout(height=total_rows * 200, width=1000, title_text="Distribution of rating criteria")
        return fig

