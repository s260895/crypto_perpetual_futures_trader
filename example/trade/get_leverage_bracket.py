from binance_f import RequestClient
from binance_f.constant.test import *
from binance_f.base.printobject import *
from binance_f.model.constant import *

g_api_key =    'uvXEaR3udbF8FolqY4ojcHmOxWCpNJmNaH4NKaHy1vYHiwGr4l3vs2TQmavcLG2E'
g_secret_key = 'FGkNqyvNUS4nmgbsebZj9Hrn60p7Sqk6WyMMqcXbq4TNFL4pBu89iG8dlUWYUody'
request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)
result = request_client.get_leverage_bracket()
PrintMix.print_data(result)
