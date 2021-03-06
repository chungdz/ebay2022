import pandas as pd
from datetime import datetime
import time
import json
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm
tqdm.pandas()

def lldis(lat1, lat2, lon1, lon2):
     
    # The math module contains a function named
    # radians which converts from degrees to radians.
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
      
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
 
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371
      
    # calculate the result
    return(c * r)

def add_func(row):
    acct = row['payment_datetime']
    dd = row['delivery_date']
    arr = acct.split()
    cdate = datetime.strptime(arr[0], "%Y-%m-%d")
    ddate = datetime.strptime(dd, "%Y-%m-%d")
    delta = ddate - cdate
    
    acc_hour = int(arr[1][:2])
    tz = int(arr[1][-6:-3])
    if cdate.month >= 3 and cdate.month <= 11:
        isdst = 0
    else:
        isdst = 1
        
    return delta.days, acc_hour, tz, isdst

def cal_dis(row):
    sender_tz = row['sender_tz']
    sender_zip = str(row['item_zip'])
    re_zip = str(row['buyer_zip'])
    
    sender_info = zip_info[sender_zip]
    re_info = zip_info[re_zip]
    
    if re_info['tz'] != False:
        tz_dis = re_info['tz'] - sender_tz - int(row['isdst'])
    else:
        tz_dis = 0
    
    if sender_info['lat'] != False and re_info['lat'] != False:
        dis = int(lldis(sender_info['lat'], re_info['lat'], sender_info['lng'], re_info['lng']) / 1000)
    else:
        dis = -1
    
    if sender_info['city'] != False and re_info['city'] != False:
        if sender_info['city'] == re_info['city']:
            cross_city = 0
        else:
            cross_city = 1
    else:
        cross_city = -1
    
    if sender_info['state'] != False and re_info['state'] != False:
        if sender_info['state'] == re_info['state']:
            cross_state = 0
        else:
            cross_state = 1
    else:
        cross_state = -1
    
    return tz_dis, dis, cross_city, cross_state

dict1 = {
    'B2C': 0,
    'C2C': 1
}

dict2 = {
    'PACKAGE_THICK_ENVELOPE': 0,
    'NONE': 1,
    'LETTER': 2,
    'LARGE_ENVELOPE': 3,
    'LARGE_PACKAGE': 4,
    'EXTRA_LARGE_PACKAGE': 5,
    'VERY_LARGE_PACKAGE': 6
}

print('load data')
raw = pd.read_csv('data/train.tsv', sep='\t')
zip_info = json.load(open('data/zipcode_dict.json', 'r'))
parsed = raw[['record_number', 'shipment_method_id', 'shipping_fee', 
'carrier_min_estimate', 'carrier_max_estimate', 'category_id', 
'item_price', 'quantity']]

print('parsing data')
fattr = raw.progress_apply(add_func, axis=1, result_type='expand')
raw['sender_tz'] = fattr[2]
raw['isdst'] = fattr[3]
dis_attr = raw.progress_apply(cal_dis, axis=1, result_type='expand')

parsed['bt'] = raw['b2c_c2c'].map(dict1)
parsed['package_size'] = raw['package_size'].map(dict2)
parsed['weight'] = raw['weight_units'] * raw['weight']

parsed['tz_dis'] = dis_attr[0]
parsed['dis'] = dis_attr[1]
parsed['cross_city'] = dis_attr[2]
parsed['cross_state'] = dis_attr[3]

parsed['acc_hour'] = fattr[1]
parsed['target'] = fattr[0]

print('Before trimming:', parsed.shape)
parsed = parsed[parsed['target'] >= 0]
print('After trimming:', parsed.shape)
parsed.to_csv('data/parsed_train.tsv', index=None, sep='\t')

