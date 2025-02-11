import datetime
from typing import Literal
from duckduckgo_search import DDGS


def search_duckduckgo(query: str, \
                        start_date:str=None, end_date:str=None, \
                        interval:Literal['m', 'w', 'd', 'y']=None, \
                        num_results=10):
    """
    Search DuckDuckGo

    :param query: Search query
    :param start_date: Start date for search (YYYY-MM-DD)
    :param end_date: End date for search (YYYY-MM-DD)
    :param interval: Interval for search - instead of date ranges
        - (m, y, d) for past month, year, day respectively
    :param num_results: Number of results to return
    :return: List results

    """
    ddg_args = {}
    if start_date and end_date:
        s_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        e_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        if s_date > e_date:
            raise ValueError("Start date must be before end date")
        if interval:
            raise ValueError("Interval is not supported for date range searches")
        ddg_args = {'timelimit': f'{start_date}..{end_date}'}

    ddgs = DDGS()
    results = []
    if interval:
        if interval not in ['m', 'w', 'd', 'y']:
            raise ValueError("Interval must be one of: m, w, d, y")
        ddg_args = {'timelimit': interval}

    for result in ddgs.text(query, max_results=num_results, **ddg_args):
        results.append({
            'title': result['title'],
            # 'link': result['href'],
            'snippet': result['body']
        })

    return results

# if __name__ == '__main__':
#     query = lambda ticker: f'{ticker} stock news'
#     start_date = '2023-01-01'
#     end_date = '2023-01-31'
#     # interval = 'm'
#     num_results = 5
#     results = search_duckduckgo(query('APPL'), start_date=start_date, end_date=end_date, num_results=num_results)
#     print(json.dumps(results, indent=2))