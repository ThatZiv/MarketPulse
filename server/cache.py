# Allows caching to have a single import
from flask_caching import Cache
cache = Cache(config={'CACHE_TYPE': 'SimpleCache'})
