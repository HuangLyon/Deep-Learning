import tensorflow as tf
import re
import os
import string
from pprint import pprint
import pandas as pd
import numpy as np
import gdown
from opencc import OpenCC
from ckiptagger import data_utils
from ckiptagger import WS
from ckiptagger import POS
from ckiptagger import NER
from ckiptagger import construct_dictionary
from pathlib import Path
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Load data from a JSON file
df = pd.read_json('XdKqXfGIP5A.json', lines = True)

# Initialize OpenCC to convert Simplified Chinese to Traditional Chinese (Taiwan usage)
cc = OpenCC('s2twp')

# Convert each text entry in the dataset to Traditional Chinese
for i in range(len(df)):
  df['text'][i] = cc.convert(df['text'][i])
df.head(5)

# Initialize stopwords list
STOPWORDS = []
# Load stopwords from file
with open('stopword.txt', 'r', encoding='utf-8') as file:
  for data in file.readlines():
    data = data.strip()
    STOPWORDS.append(data)
    STOPWORDS.append('\n')
    STOPWORDS.append(' ')
    STOPWORDS.append('。\n')


# Set up CKIP path for saving model data
path = str(Path.home()) + '/ckip/'

# If CKIP data folder doesn't exist, create it and download required data
if not os.path.exists(path):
  os.mkdir(path)
  data_utils.downlaod_data_gdown(path)
  

CKIP_WS = WS(path + "/data", disable_cuda=False)
CKIP_POS = POS(path + "/data", disable_cuda=False)
CKIP_NER = NER(path + "/data", disable_cuda=False)

seg = CKIP_WS(df['text'])

clear_word = []

for i in range(len(seg)):
  for x in seg[i]:
    if x not in STOPWORDS and x != '\n':
      clear_word.append(x)
      

def count_segment_freq(seg_list):
  seg_df = pd.DataFrame(seg_list,columns=['text'])
  seg_df['count'] = 1
  sef_freq = seg_df.groupby('text')['count'].sum().sort_values(ascending=False)
  sef_freq = pd.DataFrame(sef_freq)
  return sef_freq

# Get frequency table of segmented words
sef_freq = count_segment_freq(clear_word)
sef_freq.head()


font_path = './kaiu.ttf'
wc = WordCloud(
  background_color='black',        #   背景顏色
  max_words=200,                   #   最大分詞數量
  mask=None,                       #   背景圖片
  max_font_size=None,              #   顯示字體的最大值
  font_path=font_path,             #   若為中文則需引入中文字型(.TTF)
  random_state=None,               #   隨機碼生成各分詞顏色
  prefer_horizontal=0.9)           #   調整分詞中水平和垂直的比例


clear_word =' '.join(clear_word )
wc.generate(clear_word)

import matplotlib.pyplot as plt
plt.imshow(wc)
plt.axis('off')
plt.show()
wc.to_file('word_cloud.png')

# Search for a word in the original text entries
w = input('請輸入要搜尋的字:')
for s in range(len(df)):
  if w in df['text'][s]:
    print(df['text'][s])