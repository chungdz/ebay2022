import pandas as pd
import json
from tqdm import tqdm
from tzwhere import tzwhere

zip_code = json.load(open('data/zip_code_list.json', 'r'))
tz_dict ={
    'America/Adak': -10,
    'America/Anchorage': -9,
    'America/Asuncion': -4,
    'America/Boise': -7,
    'America/Chicago': -6,
    'America/Costa_Rica': -6,
    'America/Denver': -7,
    'America/Detroit': -5,
    'America/Guayaquil': -5,
    'America/Indiana/Indianapolis': -5,
    'America/Indiana/Knox': -6,
    'America/Indiana/Marengo': -5,
    'America/Indiana/Petersburg': -5,
    'America/Indiana/Tell_City': -6,
    'America/Indiana/Vevay': -5,
    'America/Indiana/Vincennes': -5,
    'America/Indiana/Winamac': -5,
    'America/Juneau': -9,
    'America/Kentucky/Louisville': -5,
    'America/Kentucky/Monticello': -5,
    'America/La_Paz': -4,
    'America/Lima': -5,
    'America/Los_Angeles': -8,
    'America/Managua': -6,
    'America/Menominee': -6,
    'America/New_York': -5,
    'America/Nome': -9,
    'America/North_Dakota/Beulah': -6,
    'America/North_Dakota/Center': -6,
    'America/North_Dakota/New_Salem': -6,
    'America/Ojinaga': -7,
    'America/Panama': -5,
    'America/Phoenix': -7,
    'America/Puerto_Rico': -4,
    'America/Santo_Domingo': -4,
    'America/Sitka': -9,
    'America/St_Thomas': -4,
    'America/Tegucigalpa': -6,
    'America/Toronto': -5,
    'America/Winnipeg': -6,
    'Asia/Hong_Kong': 8,
    'Asia/Karachi': 5,
    'Asia/Riyadh': 3,
    'Asia/Seoul': 9,
    'Asia/Singapore': 8,
    'Asia/Tokyo': 9,
    'Australia/Darwin': 9.5,
    'Australia/Melbourne': 10,
    'Australia/Sydney': 10,
    'Europe/Berlin': 1,
    'Europe/London': 0,
    'Pacific/Guam': 10,
    'Pacific/Honolulu': -10,
    'Pacific/Pago_Pago': -11,
    'Pacific/Saipan': 10,
    'America/Montreal': -5,
    None: False
}

ref_zip = pd.read_csv('data/uszips.csv')
ref_zip['zip'] = ref_zip['zip'].astype(str).str.zfill(5)

ref_dict = {}
for index, row in tqdm(ref_zip.iterrows()):
    cur_key = str(row['zip'])
    if cur_key in ref_dict:
        continue
    ref_dict[cur_key] = {}
    ref_dict[cur_key]['lat'] = row['lat']
    ref_dict[cur_key]['lng'] = row['lng']
    ref_dict[cur_key]['city'] = row['city']
    ref_dict[cur_key]['state'] = row['state_id']
    ref_dict[cur_key]['tz'] = row['timezone']

ref_zip2 = pd.read_csv('data/US.txt', header=None, sep='\t')
tz = tzwhere.tzwhere()
for index, row in tqdm(ref_zip2.iterrows()):
    cur_key = str(row[1])
    if cur_key in ref_dict:
        continue
    ref_dict[cur_key] = {}
    ref_dict[cur_key]['lat'] = row[9]
    ref_dict[cur_key]['lng'] = row[10]
    ref_dict[cur_key]['city'] = row[2]
    ref_dict[cur_key]['state'] = row[4]
    ref_dict[cur_key]['tz'] = tz.tzNameAt(ref_dict[cur_key]['lat'], ref_dict[cur_key]['lng'])

cleaned_zipcode = {}
cleaned_zipcode['nan'] = {
    'country': 0,
    'lat': False,
    'lng': False,
    'city': False,
    'state': False,
    'tz': False
}

for zc in tqdm(zip_code):
    zc = str(zc)
    if zc in cleaned_zipcode:
        continue
    
    if zc in ref_dict:
        nzc = zc
    elif '-' in zc and zc.split('-')[0] in ref_dict:
        nzc = zc.split('-')[0]
    elif len(zc) == 9 and zc[:5] in ref_dict:
        nzc = zc[:5]
    else:
        if '-' in zc:
            nzc = zc.split('-')[0]
        else:
            nzc = zc
        
        if nzc.isdigit():
            country = 2 + len(nzc)
        else:
            country = 2
        
        cleaned_zipcode[zc] = {
            'true_zip': nzc,
            'country': country,
            'lat': False,
            'lng': False,
            'city': False,
            'state': False,
            'tz': False
        }
        continue
    
    cleaned_zipcode[zc] = {
        'true_zip': nzc,
        'country': 1,
        'lat': ref_dict[nzc]['lat'],
        'lng': ref_dict[nzc]['lng'],
        'city': ref_dict[nzc]['city'],
        'state': ref_dict[nzc]['state'],
        'tz': tz_dict[ref_dict[nzc]['tz']]
    }

unseened_data = []
for k, v in cleaned_zipcode.items():
    if v['country'] == 7:
        unseened_data.append(k)
print('Before estimate zipcode', len(unseened_data))

left_data = []
for k, v in cleaned_zipcode.items():
    if v['country'] == 7:
        new_zip = v['true_zip'][:3] + '01'
        if new_zip in ref_dict:
            cleaned_zipcode[k] = {
                'true_zip': new_zip,
                'country': 1,
                'lat': ref_dict[new_zip]['lat'],
                'lng': ref_dict[new_zip]['lng'],
                'city': ref_dict[new_zip]['city'],
                'state': ref_dict[new_zip]['state'],
                'tz': tz_dict[ref_dict[nzc]['tz']]
            }
        else:
            left_data.append(k)

print('After:', len(left_data))

unseened_data2 = []
all_tz = set()
for k, v in cleaned_zipcode.items():
    if v['country'] == 1:
        if not v['tz'] or v['tz'] is None: 
            unseened_data2.append(k)
        else:
            all_tz.add(v['tz'])
print('US Code with no timezone:', len(unseened_data2), 'all time zone:', all_tz)

json.dump(cleaned_zipcode, open('data/zipcode_dict.json', 'w'))

