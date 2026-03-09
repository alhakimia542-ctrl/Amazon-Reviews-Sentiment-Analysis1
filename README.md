# 📊 مشروع تحليل المشاعر لمراجعات أمازون 
## Amazon Reviews Sentiment Analysis - التقرير الشامل

---

## 📋 نظرة عامة على المشروع
مشروع متكامل لتحليل المشاعر (Sentiment Analysis) باستخدام مراجعات أمازون، يبدأ من تحميل البيانات وحتى بناء تطبيق تفاعلي. قمنا بتجربة نماذج تقليدية ومتقدمة للوصول إلى أفضل أداء.

---

## 🎯 المراحل التفصيلية للمشروع

### المرحلة 1 إعداد البيئة وتحميل البيانات
```python
import kagglehub
import bz2
import pandas as pd
import os

# تحميل مجموعة البيانات من Kaggle
path = kagglehub.dataset_download(bittlingmayeramazonreviews)
print(fتم التحميل في {path})

# الملفات المستلمة
# - train.ft.txt.bz2 (بيانات التدريب)
# - test.ft.txt.bz2 (بيانات الاختبار)
```

ما تم إنجازه
- ✅ استخدام KaggleHub لتحميل 3 ملايين مراجعة أمازون
- ✅ فهم تنسيق البيانات `__label__X` حيث X=1 سلبي، X=2 إيجابي
- ✅ التعامل مع الملفات المضغوطة BZ2

---

### المرحلة 2 قراءة وتحليل البيانات الاستكشافي (EDA)
```python
def read_fasttext_bz2(file_path, num_lines=50000)
    data = []
    with bz2.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f
        for i, line in enumerate(f)
            if i = num_lines
                break
            parts = line.strip().split(' ', 1)
            if len(parts) == 2
                label = parts[0].replace('__label__', '')
                text = parts[1]
                data.append([label, text])
    return pd.DataFrame(data, columns=['sentiment', 'text'])

# قراءة 50,000 مراجعة
df = read_fasttext_bz2('kaggleinputamazonreviewstrain.ft.txt.bz2', 50000)
df['sentiment_label'] = df['sentiment'].map({'1' 'سلبي', '2' 'إيجابي'})
```

النتائج المستخلصة
 المقياس  القيمة 
-----------------
 مراجعات إيجابية  25,506 (51.0%) 
 مراجعات سلبية  24,494 (49.0%) 
 متوسط طول المراجعة  441 حرف 
 متوسط عدد الكلمات  80 كلمة 

---

### المرحلة 3 تحليل الكلمات الأكثر شيوعاً
```python
from collections import Counter
import re

def get_common_words(texts, n=15)
    words = []
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', ...}
    
    for text in texts[2000]
        text = re.sub(r'[^ws]', '', text.lower())
        for word in text.split()
            if word not in stop_words and len(word)  2
                words.append(word)
    return Counter(words).most_common(n)

# الكلمات الأكثر شيوعاً
positive_words = get_common_words(df[df['sentiment']=='2']['text'])
negative_words = get_common_words(df[df['sentiment']=='1']['text'])
```

الكلمات المميزة
 المراجعات الإيجابية  المراجعات السلبية 
----------------------------------------
 book (1122)  not (1712) 
 great (853)  book (1215) 
 good (648)  are (888) 
 read (505)  one (789) 
 like (492)  like (617) 

---

### المرحلة 4 التصور البياني (Visualization)
```python
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# رسم بياني 1 توزيع المشاعر
plt.pie([25506, 24494], labels=['إيجابي', 'سلبي'], autopct='%1.1f%%')

# رسم بياني 2 توزيع أطوال المراجعات
plt.hist([df[df['sentiment']=='2']['text_length'],
          df[df['sentiment']=='1']['text_length']], 
         label=['إيجابي', 'سلبي'], bins=50)

# رسم بياني 3 سحابة كلمات للمراجعات الإيجابية والسلبية
wordcloud_pos = WordCloud().generate(' '.join(positive_texts[2000]))
wordcloud_neg = WordCloud().generate(' '.join(negative_texts[2000]))
```

---

### المرحلة 5 بناء نماذج التصنيف التقليدية
```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import make_pipeline

# تجهيز البيانات
X = df['text'].values
y = df['sentiment'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

#### النموذج 1 Naive Bayes مع CountVectorizer
```python
model_nb = make_pipeline(
    CountVectorizer(max_features=10000, stop_words='english'),
    MultinomialNB()
)
model_nb.fit(X_train, y_train)
accuracy_nb = accuracy_score(y_test, model_nb.predict(X_test))
# النتيجة 85.46% (وقت التدريب 6.06 ثانية)
```

#### النموذج 2 Naive Bayes مع TF-IDF
```python
model_nb_tfidf = make_pipeline(
    TfidfVectorizer(max_features=10000, stop_words='english'),
    MultinomialNB()
)
model_nb_tfidf.fit(X_train, y_train)
accuracy_nb_tfidf = accuracy_score(y_test, model_nb_tfidf.predict(X_test))
# النتيجة 85.66% (وقت التدريب 4.29 ثانية)
```

#### النموذج 3 Logistic Regression مع TF-IDF
```python
model_lr = make_pipeline(
    TfidfVectorizer(max_features=10000, stop_words='english'),
    LogisticRegression(max_iter=1000, random_state=42)
)
model_lr.fit(X_train, y_train)
accuracy_lr = accuracy_score(y_test, model_lr.predict(X_test))
# النتيجة 87.87% (وقت التدريب 3.42 ثانية)
```

---

### المرحلة 6 مقارنة النماذج التقليدية
 النموذج  الدقة  وقت التدريب 
-----------------------------
 Naive Bayes  85.46%  6.06 ثانية 
 NB + TF-IDF  85.66%  4.29 ثانية 
 Logistic Regression  87.87%  3.42 ثانية 

تقرير التصنيف المفصل لأفضل نموذج (Logistic Regression)
```
              precision    recall  f1-score
سلبي             0.88      0.87      0.88
إيجابي           0.88      0.89      0.88
الدقة الإجمالية 87.87%
```

---

### المرحلة 7 بناء نموذج BERT المتقدم
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import pipeline
import torch

# تحميل نموذج BERT
model_name = bert-base-uncased
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# تجهيز البيانات لـ BERT
class AmazonReviewsDataset(torch.utils.data.Dataset)
    def __init__(self, texts, labels, tokenizer, max_length=128)
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx)
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length',
                                 max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids' encoding['input_ids'].flatten(),
            'attention_mask' encoding['attention_mask'].flatten(),
            'labels' torch.tensor(label, dtype=torch.long)
        }

# تدريب النموذج (استغرق 15-30 دقيقة)
training_args = TrainingArguments(
    output_dir='.results',
    num_train_epochs=3,
    per_device_train_batch_size=32,
    eval_strategy=epoch,
    save_strategy=epoch,
    load_best_model_at_end=True,
    fp16=True
)
```

نتائج BERT
- دقة عالية جداً في التنبؤ (تقديرياً 95%)
- ثقة في التنبؤات تصل إلى 99.9% للمراجعات الواضحة
- فهم ممتاز للسياق والمراجعات المحايدة

---

### المرحلة 8 إنشاء تطبيق تفاعلي لتحليل المشاعر 🎨
```python
import ipywidgets as widgets
from IPython.display import display
from transformers import pipeline

# تحميل نموذج BERT المدرب
sentiment_pipeline = pipeline(
    sentiment-analysis,
    model=.bert_sentiment_model,
    tokenizer=.bert_sentiment_model