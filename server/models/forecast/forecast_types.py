from typing import *

DataForecastType: TypeAlias = List[float]
DatasetType: TypeAlias = List[Dict[Union[Literal["close"] \
                                         , Literal["open"] \
                                         , Literal["high"] \
                                         , Literal["low"] \
                                         , Literal["volume"]\
                                         , Literal["sentiment_data"]\
                                         , Literal["news_data"]], DataForecastType]]

ForecastSeriesType: TypeAlias = List[Dict[Union[Literal["forecast"], \
                                                   Literal["name"]], DataForecastType]]
