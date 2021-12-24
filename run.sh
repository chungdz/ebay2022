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


python -m model.xgb_train
python -m model.xgb_quiz
