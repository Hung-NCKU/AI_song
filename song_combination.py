import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

directory = "datagame-2023"
files = os.listdir(directory)

label_encoder = LabelEncoder()

data = {}
#數值化
for file in files:
    if file.endswith(".parquet"):
        parquet_file_path = directory + '/' + file
        df = pd.read_parquet(parquet_file_path)
        
        for column in df:
            if not isinstance(df[column][0], (int, float)):
                df[column] = label_encoder.fit_transform(df[column])
        
        data[file]=df
data['meta_song_composer.parquet'] = pd.merge(data['meta_song_composer.parquet'], data['meta_song_genre.parquet'], on='song_id', how='inner')
data['meta_song_composer.parquet'] = pd.merge(data['meta_song_composer.parquet'], data['meta_song_lyricist.parquet'], on='song_id', how='inner')
data['meta_song_composer.parquet'] = pd.merge(data['meta_song_composer.parquet'], data['meta_song_producer.parquet'], on='song_id', how='inner')
data['meta_song_composer.parquet'] = pd.merge(data['meta_song_composer.parquet'], data['meta_song_titletext.parquet'], on='song_id', how='inner')
data['meta_song_composer.parquet'] = pd.merge(data['meta_song_composer.parquet'], data['meta_song.parquet'], on='song_id', how='inner')
data['meta_song_composer.parquet'] = data['meta_song_composer.parquet'].drop_duplicates()
data['meta_song_composer.parquet'].to_parquet('combination.parquet', index=False)