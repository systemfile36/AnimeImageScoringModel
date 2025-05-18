import numpy as np
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
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

def get_log_quantile_transformer(
        bookmarks: np.ndarray, views: np.ndarray, n_quantities: int = 10000
) -> tuple[QuantileTransformer, QuantileTransformer, QuantileTransformer]:
    """
    Get QuantileTranformer of log(1+x) bookmarks and views and CTR (bookmarks/(views + epsilon))
    """

    # log(1 + x)
    log_bookmarks = np.log1p(bookmarks)
    log_views = np.log1p(views)

    # bookmarks views ratio
    ctr = bookmarks / (views + 1e-5)
    log_ctr = np.log1p(ctr)

    return (
        QuantileTransformer(n_quantiles=n_quantities, output_distribution="uniform").fit(log_bookmarks.reshape(-1, 1)),
        QuantileTransformer(n_quantiles=n_quantities, output_distribution="uniform").fit(log_views.reshape(-1, 1)),
        QuantileTransformer(n_quantiles=n_quantities, output_distribution="uniform").fit(log_ctr.reshape(-1, 1))
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
    # operand between `datetime.date`
    age_days = (datetime.now().date() - upload_date)

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

def score_weighted_ctr_log_quantile_time_decay_scaled(
        bookmarks: np.ndarray, views: np.ndarray, upload_date: np.ndarray,
        alpha: float = 0.6, beta: float = 0.2, gamma: float = 0.2,
        qt_bookmarks: QuantileTransformer | None = None,
        qt_views: QuantileTransformer | None = None,
        qt_ctr: QuantileTransformer | None = None,
        n_quantities: int = 10000,
        time_decay_method: str = "sqrt",  # ["sqrt", "log", "logistic"]
        time_decay_lambda: float = 0.2
) -> np.ndarray:
    """
    Compute weighted score from bookmarks, views, and CTR with time decay and quantile normalization.

    Components:
        - log(1 + x) transform
        - Quantile normalization (normal or uniform)
        - weighted average: alpha * bookmarks + beta * views + gamma * CTR
        - time decay:
            - sqrt:        1 / sqrt(1 + days)
            - log:         1 / log(1 + days)
            - logistic:    1 / (1 + exp(β * (days - t0))) [Not implemented yet]

    Parameters:
        - bookmarks: raw bookmark counts (N,)
        - views: raw view counts (N,)
        - upload_date: np.ndarray of datetime.date (N,)
        - alpha, beta, gamma: weight for bookmarks, views, ctr
        - qt_*: pretrained QuantileTransformer (or None to fit on given data)
        - n_quantities: `n_quantities` of QuantileTransformer. Default is 10000
        - time_decay_method: time decay function selector
        - time_decay_lambda: weight to time decay. score = score * (1 - lambda) + lambda * decay)

    Returns:
        - final_score: scaled score in range 1 ~ 100 (np.ndarray)
    """

    # log(1 + x)
    log_bookmarks = np.log1p(bookmarks)
    log_views = np.log1p(views)
    ctr = bookmarks / (views + 1e-5)
    log_ctr = np.log1p(ctr)

    # Quantile normalization
    if qt_bookmarks is None:
        qt_bookmarks = QuantileTransformer(n_quantiles=n_quantities, output_distribution='uniform').fit(log_bookmarks.reshape(-1, 1))
    if qt_views is None:
        qt_views = QuantileTransformer(n_quantiles=n_quantities, output_distribution='uniform').fit(log_views.reshape(-1, 1))
    if qt_ctr is None:
        qt_ctr = QuantileTransformer(n_quantiles=n_quantities, output_distribution='uniform').fit(log_ctr.reshape(-1, 1))

    norm_bookmarks = qt_bookmarks.transform(log_bookmarks.reshape(-1, 1)).flatten()
    norm_views = qt_views.transform(log_views.reshape(-1, 1)).flatten()
    norm_ctr = qt_ctr.transform(log_ctr.reshape(-1, 1)).flatten()

    # Weighted combination
    combined = alpha * norm_bookmarks + beta * norm_views + gamma * norm_ctr

    # Time decay
    age_days = (datetime.now().date() - upload_date)
    age_days = np.array([x.days for x in age_days])

    age_days = np.clip(age_days, 1, None)

    if time_decay_method == "sqrt":
        decay = 1 / np.sqrt(1 + age_days)
    elif time_decay_method == "log":
        decay = 1 / np.log1p(age_days)
    else:
        raise ValueError("Unsupported time_decay_method. Use 'sqrt' or 'log'.")

    # Apply decay
    # decayed = combined * decay

    # Apply decay
    # Apply weight to time decay to reduce influence
    decayed = combined * ((1 - time_decay_lambda) + time_decay_lambda * decay)

    # Scale to range 1 ~ 100
    score = (decayed - decayed.min()) / (decayed.max() - decayed.min()) * 99 + 1

    return score