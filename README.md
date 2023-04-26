# cs5293sp23-project2 - Cuisine Predictor
This project collects ingredients from the user and predicts cuisine type and the cuisines closest to the provided ingredients. We use 'yummly.json' data set to train and test the data. The prediction is printed directly to the screen console.
### Requirements
1. python version 3+
### Required libraries
1. Numpy
2. Pandas
3. Sklearn
4. argparse
5. sys
6. json
7. unittest
### Commands to install libraries
```
1. pipenv install numpy
2. pipenv install pandas
3. pipenv install skicit-learn
4. pipenv install argparse
5. pipenv install sys
6. pipenv install json
7. pipenv install unittest
```
## How to run
After cloning the project, use the below commands to run the application.
Command used to run the application -
```
pipenv run python project2.py --N 5 --ingredient paprika --ingredient banana --ingredient "rice krispies"
```
The application is open to give as many number of ingredients possible by the user. Make sure to put multiple word ingredients in quotes.
Command used to run the test cases - 
```
pipenv run python -m pytest
```
## Functions
1. We use ArgumentParser() to collect the user input.
2. cuisine_predictor() is heart of the project. It takes user input as parameters and returns a dictonary with the predicted Cuisine, predicted score, and the cuisines similar to it.
3. we read yummly.json data set using read_json() available in pandas library.
```
a = pd.read_json('yummly.json')
```
4. We then lemmatize the data using WordNetLemmatizer() available in nltk.stem library.
```
lemmatizer = WordNetLemmatizer()
```
5. TfidfVectorizer is chosen to be used as the vectorizer.
```
tfidf = TfidfVectorizer(ngram_range=(1,2),stop_words="english")
```
6. Uisng the vectorizer we fit_transform the lemmatized data and use train_test_split which splits the given data to a specified train and test size.
```
c = tfidf.fit_transform(b)
X_train, X_test, y_train, y_test = train_test_split(c,a['cuisine'],test_size=0.2)
```
7. LabelEncoder() is used to convert non-numerical data to numerical format.
```
le = LabelEncoder()
```
8. In this project, we use KNeighborsClasssifier() which is a predefined machine learning model used to predict the cuisines and the probability score.I have chosed this model as it has provided highest probability score for prediction when compared with other classifiers.
```
res = KNeighborsClassifier()
res.fit(X_train,Y_train)
```
9. Then the user inputs are lemmatized, vectorized and passsed to the knn model. Final cuisine is obtained by using the inverse_transform() of the predicted array which is used to convert data into a string.
```
lem = [lemmatizer.lemmatize(i) for i in l]
vect = tfidf.transform(lem)
arr = res.predict(vect)
final_cuisine = le.inverse_transform(arr)[0]
```
10. The predicted cuisine and the score are then stored in a result dictionary.
11. Now, to predict the N nearest cuisines matching the ingredients, we again use KNeighbors classsifier.The KNN model is used to determine the distances and row positions for identical meals using the N value given by the user and the vectorized ingredients. These values are used to retrieve the meal IDs and ingredients. Cosine_similarity is used to vectorize the components from user input and similar meals in order to determine the similarity.
```
cs_list = cosine_similarity(tfidf.fit_transform(similar_ingredients),tfidf.transform(user_list))
```
12. We then add the found similarity values to the result dictionary and returned to the main function in project2.py from where the reult is displayed to the console after converting into json format.
## Test Cases
We have 5 test cases which focus on the cuisine_predictor function. The first test case focuses on whether the function is returning any value. If yes, the test case is satisfied.
```
def test1(self):
    n=5
    l=['banana','rice krispies','paprica']
    result = project222.cuisine_predictor(n,l)
    self.assertIsNotNone(result,"didn't return any value")
````
Second test case checks if the returned value is os dictionary type or not.
```
def test2(self):
    n=5
    l=['banana','rice krispies','paprica']
    result = project222.cuisine_predictor(n,l)
    if (type(result)==dict):
        true = 1
    else:
        true = 0
    assert(true==1)
```
Test cases 3,4 and 5 check if the result dictionary have the keys cuisine, score and closest.
## Assumptions and bugs
1. For ingredients that are not present in the yummly.json file there is chance of discrepencies.
