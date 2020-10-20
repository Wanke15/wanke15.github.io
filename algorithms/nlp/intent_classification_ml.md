```python
import pickle
from csv import QUOTE_NONE

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC


def load_hand_select():
    with open('./data/intent_train/xxx.txt', 'r') as f:
        func_sents = [line.strip() for line in f]

    with open('./data/intent_train/xxx.txt', 'r') as f:
        rent_sents = [line.strip() for line in f]

    with open('./data/intent_train/xxx.txt', 'r') as f:
        other_sents = [line.strip() for line in f]

    with open('./data/intent_train/xxx.txt', 'r') as f:
        poi_sents = [line.strip() for line in f]

    data = [{"text": rs, "label": 0} for rs in rent_sents]
    data.extend([{"text": fs, "label": 1} for fs in func_sents])
    data.extend([{"text": pois, "label": 2} for pois in poi_sents])
    data.extend([{"text": ots, "label": 3} for ots in other_sents])

    return pd.DataFrame(data)


df = load_hand_select()
df = pd.read_csv('./data/search_intent.csv', usecols=['text', 'label'])
df = df.loc[df['label'] != 4]

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], random_state=0, test_size=0.3)

# pd.DataFrame({"text": X_train, "label": y_train}).to_csv("./data/train.csv", index=False)
# pd.DataFrame({"text": X_test, "label": y_test}).to_csv("./data/val.csv", index=False)
# exit()

# count_vect = CountVectorizer(ngram_range=(1, 5), analyzer='char')
count_vect = CountVectorizer(ngram_range=(1, 3), analyzer='char')
X_train_feat = count_vect.fit_transform(X_train).toarray()

# hash_vectorizer = HashingVectorizer(n_features=20)
# tfidf_vectorizer = TfidfVectorizer()
# X_train_feat = tfidf_vectorizer.fit_transform(X_train)

X_test_feat = count_vect.transform(X_test).toarray()
# X_test_feat = tfidf_vectorizer.transform(X_test).toarray()


# model = MultinomialNB(alpha=0.5)
# model = RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True, max_iter=None, normalize=False, random_state=None, solver='lsqr', tol=0.01)
# model = GaussianNB()
# model = LinearSVC(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                   intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#                   multi_class='ovr', penalty='l2', random_state=None, tol=0.001,
#                   verbose=0)

svm = LinearSVC()
model = CalibratedClassifierCV(svm)

# model = LogisticRegression()
# model = DecisionTreeClassifier()
# model = GradientBoostingClassifier(subsample=1)

# categorical_features_indices = [_ for _ in range(X_train_feat.shape[1])]
# model = xgb.XGBClassifier(n_jobs=-1, objective="multi:softmax", n_estimators=100, max_depth=5)

model.fit(X_train_feat, y_train)

# with open('./models/count_vect_v2.b', 'wb') as f:
#     pickle.dump(count_vect, f)
# with open('./models/intent_v2.b', 'wb') as f:
#     pickle.dump(model, f)

print(X_train_feat.shape, y_train.shape)
print(model.score(X_train_feat, y_train))

# id_to_cat = {0: 'rent_car', 1: 'function', 2: 'poi'}
id_to_cat = {0: 'xxx', 1: 'xxx', 2: 'xxx', 3: 'xxx'}
# target_names = ["xxx", "xxx", "xxx", "xxx"]
target_names = ["xxx", "xxx", "xxx", "xxx"]
# target_names = ["xxx", "xxx", "xxx"]


# print(X_test_feat.shape, y_test.shape)
print(model.score(X_test_feat, y_test))

print(df['label'].value_counts())
print(classification_report(y_test, model.predict(X_test_feat), target_names=target_names))


def myPredict(sec):
    input_feat = count_vect.transform([sec])
    # input_feat = tfidf_vectorizer.transform([sec])
    pred_cat_id = model.predict(input_feat)
    print("输入: {:<12}\t".format(sec), "\t识别意图: {}".format(id_to_cat[pred_cat_id[0]]),
          '\t识别概率: {:.5f}'.format(max(model.predict_proba(input_feat)[0])))


myPredict('测试一下')


```
