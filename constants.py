RATING_CRITERIA_COLS = [
    'matches_prompt',        # Story matches prompt
    'story_coherence',       # Content coherence
    'event_structure',       # Event structure
    'linguistic_coherence',  # Linguistic coherence
    'grammar',               # Grammar correctness
    'style_of_writing',      # Style of writing
    'overall_quality',       # Overall quality
    'matches_topic',         # Matches Topic
    'matches_genre',         # Matches Genre
    'argument_structure',    # Argumentative Structure
    'factual_correctness',   # former Fact Check -> explicit Fact Check
    'content_scope',         # Scope
    'mentions keywords',     # Story mentions keywords
    'matches keywords',      # Story matches keywords
    'sentiment',             # Keywords mostly used correctly OR Source text matches keywords
    'action_coherence',      # Action coherence
    'feature_consistency',
    'stable_setting',
    'genre_consistency',
    'story_transparency',
    'tension_curve',         # former Event structure
    'linguistic_difference',
    'content_similarity',
    'identical_information',
    'exhaustive_information',
    'context_realization',   # Context realization
    'implicit_fact_check',   # Implicit Fact Check
    'citation_correct',      # Citation correct
    'false_gender',          # Gender changes which causes an implicit fact mistake
    'false_mood',            # Mood changes which causes an implicit fact mistake
    'readability',           # new Style of Writing
    'topic_met',             # Topic of headline met
    'main_message_met',      # Main message of headline met
    'matches_style_tone',    # Style and tone of headline matches
    'content_correctness',   # Content of headline is correct
    'comphrehension',        # Headline is comphrehensible
    'spelling_and_grammar',  # Headline grammar
    'info_consistency',      # Headline's infos are consistent
    'matches_fandom'         # Story matches fandom
]
OVERALL_CRITERIA_COLUMN = 'overall_quality'
AVG_RATING_COL = 'avg_rating'
EXCLUDE_USER_IDS = [40, 41, 12803, 15377, 71096]
GUEST_USER_ID = 40
COLS = {
    'turn': 'turn',
    'listvalue': 'listvalue',
    'model': 'model',
    'user': 'user',
    'story': 'story',
    'prompt': 'prompt',
    'title': 'title',
    'summary': 'summary',
    'beginning': 'beginning',
    'ending': 'ending',
    'topics': 'topics',
    'contexts': 'contexts',
    'story_hash': 'story_hash',
    'story_id': 'story_id',
    'result': 'result',
    'keywords': 'keywords',
    'duration_generation': 'duration_generation',
    'dur_score': 'dur_score',
    'amount_fallbacks': 'amount_fallbacks',
    'comment': 'comment',
    'positive_comment': 'positive_comment',
    'positive_example': 'positive_example',
    'positive_comment_topic': 'positive_comment_topic',
    'negative_comment': 'negative_comment',
    'negative_example': 'negative_example',
    'negative_comment_topic': 'negative_comment_topic'
}

TURNS = ['turnx92nf', 'turnx93','turnx96nf','turnx97', 'turnx98nf', 'turnx100nf', 'turnx101', 'turnx102nf', 'turnx103']


