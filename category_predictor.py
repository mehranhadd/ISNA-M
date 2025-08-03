import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_csv('Clean_ISNA_Dataset.csv')
df = df.astype({'Content': 'str', 'Category': 'category'})

# Now we use Bag of Words to feed our data to the ML algorithm.
# Convert our data to numpy arrays
x = np.array(df.iloc[:, 0].values, ndmin=2).T
y = np.array(df.iloc[:, 1].values, ndmin=2).T

# Use CountVectorizer to assign frequent words to 1 and non-frequent to 0.
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(df.Content).toarray()

# Apply one_hot_encoder on our labels
enc = OneHotEncoder(sparse_output=False)
Y = enc.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.4, random_state=42, shuffle=True)

# Now try to find best value of hyperparameter 'n_estimators' of the RandomForestClassifier.
t = [10, 25, 50, 100, 500, 1000]
mcm = {}
for n in t:
    rfc = RandomForestClassifier(n_estimators = n, random_state=42)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    mcm[n] = classification_report(y_test, y_pred)

for key, value in mcm.items():
    print(key, value)
    print('------------------')