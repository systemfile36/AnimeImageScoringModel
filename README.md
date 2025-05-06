# AnimeImageScoring Model - Temporary

A CNN-based TensorFlow model for predicting the popularity and aesthetic score of subculture character images using metadata-aware learning.

I try to make two model

1. Multi-task model with metadata predict + uncertainty-weighted loss
2. (1) + embeding metadata
3. (3) + self attention

## Experiments - Temporary

Select backbone model (Feature extractor)

1. Pre-trained ResNet152 (ImageNet)
2. Pre-trained EfficientNetB7 (ImageNet)
3. Original ResNet152
4. Customized ResNet

Compare models with embeding

1. With no embeding
2. With embeding AI type and sanity level
3. With embeding tags
4. With embeding both of (2), (3)

Compare models with method of embeding 

1. Multi-task
2. Multi-task with embeding predicted metadata
3. (2) + self-attention

## Requirements and Environment 

This project require Tensorflow 2.17.0 or above.

See `requirements.txt`. 

Or use Docker Image

```bash
> cd /path_to_repository
> docker build -t anime-scoring-env
> docker run --gpus all -it --rm -v /path_to_dataset_directory:/data \
    anime-scoring-env bash
> 
```

## Dataset Structure

(Now, in progress. Can be updated any time)

This project uses following directory structure for dataset.
Image files and metadata files are located in sub directory named first two characters of Pixiv artwork's id.

```plaintext
dataset/
    - .database/
        - metadata.sqlite3
    - 28/
        - 28........png
        - 28........png.json
        - ......
    - 29/
        - 29.........png
        - 29.........jpg.json
        - ......
    - 80/
        - 80.........png
        - 80.........jpg.json
        - ......
```

All of images file must be PNG format. 
metadata files must have same name with image file.

metadata file's structure is same to output of [`gallery-dl`](https://github.com/mikf/gallery-dl) using `--write-metadata` option. (Using `PixivSearchExtractor`)
See below.

```json
// Pixiv metadata format that gallery-dl generated 
{
    "id": 00000000,
    "title": "......",
    "type": "illust",
    "caption": "......",
    "restrict": 0,
    "user": {
        "id": 000000,
        "name": "..",
        "account": "...",
        "profile_image_urls": {
            "medium": "https://....jpg"
        },
        "is_followed": false,
        "is_accept_request": false
    },
    "tags": [
        "オリジナル",
        "...",
    ],
    "tools": [
        "Photoshop"
    ],
    "create_date": "1970-01-01T00:00:00+09:00",
    "page_count": 1,
    "width": 512,
    "height": 512,
    "sanity_level": 2,
    "x_restrict": 0,
    "series": null,
    "total_view": 5000,
    "total_bookmarks": 100000,
    "is_bookmarked": false,
    "visible": true,
    "is_muted": false,
    "illust_ai_type": 0,
    "illust_book_style": 0,
    "request": null,
    "num": 0,
    "date": "1970-01-01 00:00:00",
    "rating": "General",
    "suffix": "",
    "search": {
        "word": "......",
        "sort": "popular_desc",
        "target": "partial_match_for_tags",
        "date_start": null,
        "date_end": null
    },
    "category": "pixiv",
    "subcategory": "search",
    "url": "https://i.pximg.net/img-original/img/.....jpg",
    "date_url": "1970-01-01 00:00:00",
    "filename": "00000000_p0.jpg",
    "extension": "jpg"
}
```

SQLite database file must be contains table `illusts`

`illusts` table must be contains following columns

```sql
CREATE TABLE illusts (
    filename TEXT PRIMARY KEY,
    sanity_level INTEGER,
    total_bookmarks INTEGER
    total_view INTEGER,
    tags TEXT, 
    tag_character TEXT,
    illust_ai_type INTEGER,
    page_count INTEGER,
    date TIMESTAMP
)
```

`filename` is filename of image file without extension. (e.g., `00000000_p1`)

`sanity_level` represent rating. See below 

| value | description |
| :---- | :---------- |
| 2     | General. Safe for all age. Nothing sexualized. |
| 4     | Sensitive. Contain mildly erotic. minor sexuality |
| 6     | NSFW. Contain nudity, exposed genitals, explicit secuality |

`illust_ai_type` is AI type of image. See below 

| value | description |
| :---- | :---------- |
| 0     | Not AI-generated. Human-drawn illust |
| 1     | Unknown. It's unclear whether image was AI-generated or human-drawn |
| 2     | AI-generated. |

We can't sure whether the image was AI-generated or human-drawn by `illust_ai_type`. 
`illust_ai_type=0` and `illust_ai_type=2` is explicit. But `illust_ai_type=1` is unclear. 
So I will use following query to check ai_type of image.

```sql
SELECT ai_type, COUNT(*) as num
FROM (
    SELECT 
        CASE 
            WHEN illust_ai_type = 2 
              OR tags LIKE '%AIイラスト%' 
              OR tags LIKE '%Diffusion%' 
              OR tags LIKE '%Novel%' 
              OR tags LIKE '%midjourney%' 
            THEN 2 
            ELSE 1 
        END AS ai_type, total_bookmarks, total_view
    FROM illusts
)
GROUP BY ai_type;
```

`ai_type=1` represent the image file is human-drawn. 
`ai_type=2` represent the image file is AI-generated.

`total_bookmarks` and `total_views` will be used for Y-value of model.

`tags` is comma-seperated Japanese tag list of Pixiv artworks. 

`tag_character` is Danbooru character tag name mapped from Pixiv tags.
See `CreateCharacterTagsSqlite.py` and `AddTagCharacterColumn.py` in [`PixivDatasetUtilsForImageScoring` repository](https://github.com/systemfile36/PixivDatasetUtilsForImageScoring)

`page_count` is count of pages. 

`date` is the date have been uploaded illust.

## Data preprocessing 

### Preprocess for Score - Scaling, Normalize

I will use `total_bookmarks` and `total_views` value from dataset as Y-value of model. 

Both are long-tail data. 
So I will use log scale, nomalize, weighted average. 

It's `score_weighted_log_average` and `score_weighted_log_average_scaled`

```latex
f(x_b, x_v) = \alpha\cdot 
\frac{\log(1 + x_b)-\min(\log(1+x_{b_j}))}
{\max(\log(1+x_{b_j}))-\min(\log(1+x_{b_j}))}+
(1-\alpha)\cdot
\frac{\log(1+x_v)-\min(\log(1+x_{v_j}))}
{\max(\log(1+x_{v_j}))-\min(\log(1+x_{v_j}))}
```

`x_b` and `x_v` mean `total_bookmarks` and `total_view` of a sample.

`x_{b_j}` and `x_{v_j}` mean `total_bookmarks` and `total_view`of entire samples.

```py
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

    You should provide scalers when 
    """

    # log transform 
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
```

(korean)

하지만 위의 점수 산정 방법은, 이미지가 업로드된 시간을 고려하지 못한다. 즉, 올라온지 오래된 이미지일 수록 더 유리하다는 것이다. 

왜냐하면 유저에게 노출될 기회와 북마크를 받을 기회가 더욱 많기 때문이다. 따라서, 가중 평균에 조회수 대비 북마크 수(`score_bookmarks_ratio`)를 포함한 버전을 만들자. 

