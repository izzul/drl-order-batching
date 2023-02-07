import pandas as pd
from datetime import datetime
import re

def string_to_list(string):
    string = string[1:-1]
    string = re.findall('\(.+?\)', string)
    i = 0
    while i < len(string):
        string[i] = re.findall('[0-9]+', string[i])
        i += 1
    return string

def convert_data(path):
    initial_time = datetime.timestamp(datetime.strptime("01/03/2022 08:00:00", "%d/%m/%Y %H:%M:%S"))
    src_data = pd.read_csv(path, index_col='Order ID')
    # src_data['Order ID'].rename('OrderID', axis=1, inplace=True)
    src_data['skuIDlist'] = src_data.apply(lambda x: [int(skuid[0]) for skuid in string_to_list(x['Order List'])], axis=1)
    src_data['skuQuantityList'] = src_data.apply(lambda x: [int(skuid[1]) for skuid in string_to_list(x['Order List'])], axis=1)
    src_data['comp'] = 'MIO'
    src_data['nItems_ptg'] = src_data.apply(lambda x: sum(x['skuQuantityList']), axis=1)
    src_data['nItems_gtp'] = 0
    src_data['arrival_time'] = src_data.apply(lambda x:
                                              datetime.timestamp(datetime.strptime(x['Created Time'], "%d/%m/%Y %H:%M:%S")) -\
                                              initial_time, axis=1)
    src_data['cutoff_time'] = src_data.apply(lambda x:
                                             datetime.timestamp(datetime.strptime(x['Due Time'], "%d/%m/%Y %H:%M:%S")) -
                                             datetime.timestamp(datetime.strptime(x['Created Time'], "%d/%m/%Y %H:%M:%S")), axis=1)
    src_data.drop(['Created Time', 'Order List', 'Total Item', 'Due Time'], axis=1, inplace=True)
    src_data.index.names = ['orderID']
    return src_data

new_data = convert_data(r'/mnt/9A248A0E2489EE17/Users/aizzu/Documents/Works/AIZ/Tikno/new_code/orderFile/orderFile09.csv')
new_data.to_csv('OrderFile09.csv')
