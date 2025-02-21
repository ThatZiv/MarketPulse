from typing import *

DataForecastType: TypeAlias = List[float]
DatasetType: TypeAlias = List[Dict[Union[Literal["close"] \
                                         , Literal["open"] \
                                         , Literal["high"] \
                                         , Literal["low"] \
                                         , Literal["volume"]], DataForecastType]]

ForecastSeriesType: TypeAlias = List[Dict[Union[Literal["forecast"], \
                                                   Literal["name"]], DataForecastType]]
