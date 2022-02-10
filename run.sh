mkdir data
cd data
wget https://download.geonames.org/export/zip/US.zip
unzip US.zip
wget https://simplemaps.com/static/data/us-zips/1.79/basic/simplemaps_uszips_basicv1.79.zip
unzip simplemaps_uszips_basicv1.79.zip

cd ..
python -m process_data.collect_zipcode
python -m process_data.set_zipcode_info
python -m process_data.parse_train
python -m process_data.split_train
python -m process_data.parse_quiz
python -m process_data.one_hot_encode
python -m process_data.split_train --target parsed_train_cat.tsv --filename=subtrain_cat

python -m model.xgb_train
python -m model.xgb_quiz

python -m model.LightGBM_train
python -m model.LightGBM_quiz

python -m modules.cat_train
python -m modules.cat_quiz

python -m model.nn_train_single.py --starti=4
python -m model.nn_train_single.py --starti=5
python -m model.nn_train_single.py --starti=6
python -m model.nn_train_single.py --starti=7
python -m model.nn_train_single.py --starti=8
python -m model.nn_train_single.py --starti=9
python -m model.nn_train_single.py --starti=10

python -m modules.fnn_train
python -m modules.fnn_quiz

python -m process_data.create_l2_data
python -m process_data.split_train --target sl_data/parsed_train.tsv --filename=sl_data/subtrain

python -m modules.l2_regre_train
python -m modules.l2_regre_quiz
