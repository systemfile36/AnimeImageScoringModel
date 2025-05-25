# Export dataset wrapper classes 
from .dataset_wrappers import DatasetWrapper
from .dataset_wrappers import DatasetWithMetaWrapper
from .dataset_wrappers import DatasetWithMetaAndTagCharacterWrapper
from .dataset_wrappers import DatasetWrapperForScoreClassification
from .dataset_wrappers import DatasetWrapperForRatingClassification
from .dataset_wrappers import DatasetWrapperForAiClassification

from .dataset import load_all_character_tags, load_records, load_all_character_tags_from_json, save_all_character_tags_as_json

from .preprocessing import score_bookmarks_ratio
from .preprocessing import score_weighted_log_average, score_weighted_log_average_scaled
from .preprocessing import score_weighted_log_average_time_decay
from .preprocessing import score_weighted_ctr_log_quantile_time_decay_scaled
from .preprocessing import get_log_minmax_scaler
from .preprocessing import get_log_quantile_transformer

SCORE_PREPROCESS_MAP = {
    'score_bookmarks_ratio': score_bookmarks_ratio,
    'score_weighted_log_average': score_weighted_log_average,
    'score_weighted_log_average_scaled': score_weighted_log_average_scaled, 
    'score_weighted_log_average_time_decay': score_weighted_log_average_time_decay
}