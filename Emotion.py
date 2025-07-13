from ckiptagger import WS, POS
from pathlib import Path

path = str(Path.home()) + '/ckip/'
CKIP_WS = WS(path + "/data", disable_cuda=False) # 載入斷詞模型
CKIP_POS = POS(path + "/data", disable_cuda=False) # 載入詞性標記模型


text = "我不知道自己怎麼了，每天早上醒來都覺得好累，像是完全沒有力氣面對這個世界。好像什麼都沒意思，朋友約我出去我也不想去，覺得自己拖累了大家。我試著振作，但腦子裡總有一團霧，怎麼也散不開。昨天我看著窗外的天空，灰灰的，覺得它和我一樣空洞。有時候我會想，是不是我永遠都走不出這種感覺？連笑都覺得好假，好像在演戲給別人看。我真的好想回到以前那個開心的自己，但現在的我，連怎麼快樂都忘了。"
word_sentence_list = CKIP_WS([text])
pos_sentence_list = CKIP_POS(word_sentence_list)

print("斷詞結果：", word_sentence_list)
print("詞性標記結果：", pos_sentence_list)

# 假設有一個簡單的情感詞典
positive_words = {'很棒': 1, '喜歡': 1, '非常': 0.5}
negative_words = {'討厭': -1, '糟糕': -1, '累' : -1,'死' : -2,'害怕' : -1,}

def simple_sentiment_analysis(words, pos_tags):
    score = 0
    for word, pos in zip(words, pos_tags):
        if word in positive_words:
            score += positive_words[word]
        elif word in negative_words:
            score += negative_words[word]
    return "正面" if score > 0 else "負面" if score < 0 else "中性"

# 應用到斷詞結果
sentiment = simple_sentiment_analysis(word_sentence_list[0], pos_sentence_list[0])
print("情感分析結果：", sentiment)

from transformers import pipeline
sentiment_classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

result = sentiment_classifier(text)
print("情感分析結果：", result)

