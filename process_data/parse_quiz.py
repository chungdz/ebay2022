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
    acct = row['acceptance_scan_timestamp']
    arr = acct.split()
    cdate = datetime.strptime(arr[0], "%Y-%m-%d")
    acc_hour = int(arr[1][:2])
    tz = int(arr[1][-6:-3])
    
    if cdate.month >= 3 and cdate.month <= 11:
        isdst = 0
    else:
        isdst = 1
    
    payd = row['payment_datetime']
    arr2 = payd.split()
    phour = int(arr2[1][:2])
    acc_date = int(cdate.strftime("%Y%m%d"))
    if str(acc_date)[:-4] >= '1124':
        isHoliday = 1
    else:
        isHoliday = 0
        
    return acc_hour, tz, isdst, phour, acc_date, isHoliday

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
        dis = float(lldis(sender_info['lat'], re_info['lat'], sender_info['lng'], re_info['lng']) / 1000)
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
    
    return tz_dis, dis, cross_city, cross_state, sender_info['state_idx'], re_info['state_idx']

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

dict3 = {
    2: 2.20462,
    1: 1
}

print('load data')
raw = pd.read_csv('data/quiz.tsv', sep='\t')
zip_info = json.load(open('data/zipcode_dict.json', 'r'))

parsed = raw[['record_number', 'shipment_method_id', 'shipping_fee', 
'carrier_min_estimate', 'carrier_max_estimate', 'category_id', 
'item_price', 'quantity', 'declared_handling_days']]

print('parsing data')
fattr = raw.progress_apply(add_func, axis=1, result_type='expand')
raw['sender_tz'] = fattr[1]
raw['isdst'] = fattr[2]
dis_attr = raw.progress_apply(cal_dis, axis=1, result_type='expand')

svc = raw['seller_id'].value_counts()
seller_dict = {}
for sid, scounts in zip(svc.index.values, svc.values):
    seller_dict[sid] = scounts
parsed['seller_size'] = raw['seller_id'].map(seller_dict)

parsed['bt'] = raw['b2c_c2c'].map(dict1)
parsed['package_size'] = raw['package_size'].map(dict2)
parsed['weight'] = raw['weight_units'].map(dict3) * raw['weight']
parsed['shipping_units'] = parsed['shipping_fee'] / parsed['weight']

parsed['tz_dis'] = dis_attr[0]
parsed['dis'] = dis_attr[1]
parsed['cross_city'] = dis_attr[2]
parsed['cross_state'] = dis_attr[3]
parsed['sender_state'] = dis_attr[4]
parsed['receive_state'] = dis_attr[5]

parsed['acc_hour'] = fattr[0]
parsed['isNextDay'] = parsed['acc_hour'] >= 14
parsed['pay_hour'] = fattr[3]
parsed['acc_date'] = fattr[4]
parsed['isHoliday'] = fattr[5]

print('shape:', parsed.shape)
parsed.to_csv('data/parsed_quiz.tsv', index=None, sep='\t')

