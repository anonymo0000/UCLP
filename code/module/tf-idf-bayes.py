import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
#          ，            
documents = ["Buffer in overflows in PL/SQL module ",
             "in the conteudo parameter."]
labels = ["1", "2"]

# 1.      


def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d', '', text)
    return text.lower()


#           
documents = [preprocess_text(doc) for doc in documents]

# 2.   TF-IDF      
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

print(X)
# 3.  TF-IDF       
feature_names = vectorizer.get_feature_names_out()

#                TF-IDF  
tfidf_values = X.toarray()

#       ，         TF-IDF       
sorted_indices = np.argsort(-tfidf_values, axis=1)

#              TF-IDF       
for i, indices in enumerate(sorted_indices):
    doc = documents[i]
    print(f"  {i + 1}       TF-IDF   ：")
    for j in indices:
        word = feature_names[j]
        tfidf = tfidf_values[i, j]
        print(f"{word}: {tfidf}")
    print("\n")

# 4.  TF-IDF           X_modified 
modification_value = 0.1
X_modified = np.copy(tfidf_values) + modification_value

#              TF-IDF   （   ）
for i, indices in enumerate(sorted_indices):
    doc = documents[i]
    print(f"  {i + 1}       TF-IDF   （   ）：")
    for j in indices:
        word = feature_names[j]
        tfidf_modified = X_modified[i, j]
        print(f"{word}: {tfidf_modified}")
    print("\n")


# 3.      
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.2, random_state=42)


# 4.          
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train, y_train)

# 5.   
y_pred = naive_bayes_classifier.predict(X_test)

# 6.     
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#           
print("Classification Report:\n", classification_report(y_test, y_pred))
