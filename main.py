import spacy
import warnings

warnings.filterwarnings("ignore")
# nlp.add_pipe("yake")

import numpy as np
import pytextrank
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch

data = pd.read_csv('us_equities_news_dataset.csv', index_col=0)
dataCleaning = data[
    (data['provider'] == 'Seeking Alpha') | (data['provider'] == 'MarketWatch') | (data['category'] == 'opinion')].index
data.drop(dataCleaning, inplace=True)

data = data.reset_index()
data = data.drop(labels=range(2000, 82193), axis=0)

nlp = spacy.load("en_core_web_md")
nlp.add_pipe("textrank")

headlines_array = np.array(data)
headlines_list = list(headlines_array[:, 2])

# Getting the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
inputs = tokenizer(headlines_list, padding=True, truncation=True, return_tensors='pt')
outputs = model(**inputs)
print(outputs.logits.shape)
model.config.id2label

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

# Headline #Positive #Negative #Neutral
positive = predictions[:, 0].tolist()
negative = predictions[:, 1].tolist()
neutral = predictions[:, 2].tolist()

table = {'title': headlines_list,
         "Positive": positive,
         "Negative": negative,
         "Neutral": neutral}

df = pd.DataFrame(table, columns=["title", "Positive", "Negative", "Neutral"])
df['sentiment'] = df[['Positive', 'Negative', 'Neutral']].idxmax(axis=1)
data['sentiment'] = df['sentiment']

sentimentCleaning = data[(data['sentiment'] == 'Neutral')].index
data.drop(sentimentCleaning, inplace=True)


# def get_hotwords(text):
#     doc = nlp(text)
#
#     keywords = ''
#     for keyword, score in doc._.extract_keywords(n=10):
#         keywords += str(keyword) + ' '
#     return keywords
#
#
# def getSummaryPerArticle(text):
#     if text == "":
#         print(text)
#     query = get_hotwords(text)
#     doc = nlp(text)
#     query = nlp(query)
#     similarity_vector = [(sentence.similarity(query), sentence.text) for i, sentence in enumerate(doc.sents)]
#     sorted_by_similarity = sorted(similarity_vector, key=lambda x: x[0], reverse=True)
#     print(sorted_by_similarity[0])
#     return sorted_by_similarity[0][1]


def getSummaryPerArticle(text):
    doc = nlp(text)
    summary = doc._.textrank.summary(limit_phrases=15, limit_sentences=1)
    result = ''
    for sentence in summary:
        result += sentence.text
    return result


data['summary'] = data.apply(lambda row: getSummaryPerArticle((row['content'])), axis=1)

data.to_csv('summary.csv')
