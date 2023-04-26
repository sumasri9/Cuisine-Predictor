import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('wordnet')
def cuisine_predictor(number,l):
    result = {}
    a = pd.read_json('yummly.json')
    lemmatizer = WordNetLemmatizer()
    b = []
    for j in a['ingredients']:
        data = ' '.join([lemmatizer.lemmatize(i) for i in j])
        b.append(data)
    tfidf = TfidfVectorizer(ngram_range=(1,2),stop_words="english")
    c = tfidf.fit_transform(b)
    X_train, X_test, y_train, y_test = train_test_split(c,a['cuisine'],test_size=0.2)
    le = LabelEncoder()
    le.fit(a['cuisine'])
    Y_train = le.transform(y_train)
    res = KNeighborsClassifier()
    res.fit(X_train,Y_train)
    lem = [lemmatizer.lemmatize(i) for i in l]
    vect = tfidf.transform(lem)
    arr = res.predict(vect)
    final_cuisine = le.inverse_transform(arr)[0]
    result["cuisine"] = final_cuisine
    sc = np.round_(res.predict_proba(vect)[0],decimals=2)
    final_score = float(np.amax(sc))
    result["score"] = final_score
    knc = KNeighborsClassifier(n_neighbors=number)
    knc.fit(c, list(a['cuisine']))
    distances, ids = knc.kneighbors(vect, number)
    idlist = ids[0]
    similar_ingredients = []
    ingredient_list = []
    for idx in idlist:
        user_list = ' '.join(a['ingredients'].loc[idx])
        ingredient_list.append(int(a['id'].loc[idx]))
        similar_ingredients.append(user_list)
    ing_str = ' '.join(l)
    user_list = [ing_str]
    cs_list = cosine_similarity(tfidf.fit_transform(similar_ingredients),tfidf.transform(user_list))
    similarity_values = []
    for value in cs_list:
        similarity_values.append(value[0])
    cList = []
    for i in range(0, len(similarity_values)):
        tDict = {}
        tDict["id"] = ingredient_list[i]
        tDict["score"] = similarity_values[i]
        cList.append(tDict)
    result["closest"] = cList
    return result

