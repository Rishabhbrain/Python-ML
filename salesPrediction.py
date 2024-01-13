import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from io import StringIO  # Correct import for StringIO

# Copy and paste the dataset here
data = """
Age  |  Salary  |  Status
25   |  50000   |  1
30   |  60000   |  1
22   |  45000   |  0
35   |  70000   |  1
28   |  55000   |  0
40   |  80000   |  1
32   |  65000   |  0
45   |  90000   |  1
27   |  52000   |  0
38   |  75000   |  1
26   |  48000   |  0
33   |  62000   |  1
29   |  57000   |  0
42   |  82000   |  1
31   |  63000   |  0
48   |  95000   |  1
24   |  51000   |  0
36   |  72000   |  1
28   |  54000   |  0
41   |  86000   |  1
34   |  68000   |  0
46   |  92000   |  1
29   |  59000   |  0
39   |  78000   |  1
27   |  53000   |  0
37   |  74000   |  1
23   |  46000   |  0
44   |  89000   |  1
30   |  61000   |  0
49   |  97000   |  1
26   |  50000   |  0
35   |  73000   |  1
28   |  56000   |  0
43   |  88000   |  1
32   |  67000   |  0
47   |  94000   |  1
28   |  60000   |  0
40   |  81000   |  1
30   |  59000   |  0
37   |  76000   |  1
31   |  64000   |  0
50   |  99000   |  1
29   |  58000   |  0
39   |  80000   |  1
27   |  52000   |  0
38   |  77000   |  1
24   |  49000   |  0
45   |  93000   |  1
28   |  55000   |  0
36   |   73000   |  
"""

# Use StringIO correctly
df = pd.read_csv(StringIO(data), sep="|", skipinitialspace=True)

# Split the data into features (X) and target variable (y)
X = df.drop('Status', axis=1)
y = df['Status']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
model = LogisticRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:\n", classification_report_result)
