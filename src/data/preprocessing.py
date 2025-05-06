import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def score_weighted_log_average_scaled(
        bookmarks: np.ndarray, views: np.ndarray, alpha: float=0.7,
        scaler_bookmarks: MinMaxScaler | None = None, scaler_views: MinMaxScaler | None = None) -> np.ndarray:
    """
    Compute weighted log-scaled average of bookmarks and views.

    If scalers are provided, use them for transformation only (no fitting).
    If not provided, fit new scalers on the input and use them.

    Scale to range 1 ~ 100
    """
    weighted = score_weighted_log_average(bookmarks, views, alpha, scaler_bookmarks, scaler_views)

    # Scale to range 1 ~ 100
    score = (weighted - weighted.min()) / (weighted.max() - weighted.min()) * 99 + 1

    return score

def get_log_minmax_scaler(bookmarks: np.ndarray, views: np.ndarray) -> tuple[MinMaxScaler, MinMaxScaler]:
    return (
        MinMaxScaler().fit(np.log1p(bookmarks).reshape(-1, 1)),
        MinMaxScaler().fit(np.log1p(views).reshape(-1, 1))
    )

def score_weighted_log_average(
        bookmarks: np.ndarray, views: np.ndarray, alpha: float=0.7,
        scaler_bookmarks: MinMaxScaler | None = None, scaler_views: MinMaxScaler | None = None) -> np.ndarray:
    """
    Compute weighted log-scaled average of bookmarks and views.

    If scalers are provided, use them for transformation only (no fitting).
    If not provided, fit new scalers on the input and use them.

    You should provide scalers when use this for preprocessing validation(or test) data.
    """

    # log(1 + x) transform 
    log_bookmarks = np.log1p(bookmarks)
    log_views = np.log1p(views)

    if scaler_bookmarks is None:
        scaler_bookmarks = MinMaxScaler().fit(log_bookmarks.reshape(-1, 1))
    
    if scaler_views is None:
        scaler_views = MinMaxScaler().fit(log_views.reshape(-1, 1))

    # MinMax scaling : (x - x.min)/(x.max - x.min)
    # Shape transition: (N, ) -> (N, 1) -> (N, )
    norm_bookmarks = scaler_bookmarks.transform(log_bookmarks.reshape(-1, 1)).flatten()
    norm_views = scaler_views.transform(log_views.reshape(-1, 1)).flatten()

    # Weighted average
    weighted = (alpha * norm_bookmarks) + ((1 - alpha) * norm_views)

    return weighted

def score_bookmarks_ratio(bookmarks: np.ndarray, views: np.ndarray, epsilon=1e-5) -> np.ndarray:
    return bookmarks / (views + epsilon)

def score_weighted_log_average_time_decay(
        bookmarks: np.ndarray, views: np.ndarray, upload_date: np.ndarray, alpha: float=0.7, 
        scaler_bookmarks: MinMaxScaler | None = None, scaler_views: MinMaxScaler | None = None) -> np.ndarray:
    """
    Compute weighted log-scaled average of bookmarks and views with applying time decay.

    If scalers are provided, use them for transformation only (no fitting).
    If not provided, fit new scalers on the input and use them.

    Time decay : time_decay(upload_date) = 1 / log(1 + (upload_date - now).days) 
    """

    weighted = score_weighted_log_average(bookmarks, views, alpha, scaler_bookmarks, scaler_views)

    # Get timedelta `days`
    age_days = (datetime.now() - upload_date)

    # Convert datetime.timedelta to days: int
    age_days = np.array([x.days for x in age_days])

    # Get decay by f(x) = 1 / log(1 + x)
    time_decay = 1 / np.log1p(age_days)

    return weighted * time_decay

def score_weighted_log_average_time_decay_scaled(
        bookmarks: np.ndarray, views: np.ndarray, upload_date: np.ndarray, alpha: float=0.7, 
        scaler_bookmarks: MinMaxScaler | None = None, scaler_views: MinMaxScaler | None = None) -> np.ndarray:
    """
    Compute weighted log-scaled average of bookmarks and views with applying time decay.

    If scalers are provided, use them for transformation only (no fitting).
    If not provided, fit new scalers on the input and use them.

    Time decay : time_decay(upload_date) = 1 / log(1 + (upload_date - now).days) 

    Scale to range 1 ~ 100
    """

    weighted = score_weighted_log_average_time_decay(bookmarks, views, upload_date, alpha, scaler_bookmarks, scaler_views)

    # Scale to range 1 ~ 100
    score = (weighted - weighted.min()) / (weighted.max() - weighted.min()) * 99 + 1

    return score