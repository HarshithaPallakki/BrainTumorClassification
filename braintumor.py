import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import os
import cv2
import pickle

# Define the classes
classes = {"notumor" : 0, "pituitary": 1, "meningioma": 2, "glioma": 3}

# Load and preprocess the data
x = []
y = []

for cls in classes:
    pth = "C:/Users/hp/Downloads/archive/Training/" + cls
    for i in os.listdir(pth):
        img = cv2.imread(pth+"/"+i, 0)
        img = cv2.resize(img, (200,200))
        x.append(img.flatten())
        y.append(classes[cls])

x = np.array(x)
y = np.array(y)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=41)

# Create and train the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
rf_classifier.fit(x_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion_mat)
print("Accuracy:", accuracy)




pickle.dump(rf_classifier,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))