#Install Dependencies
pip install pandas scikit-learn flask joblib

#Training & Saving Model (train_model.py)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Sample trade data
trade_db = pd.DataFrame({
    "tradeID": ["T1", "T2", "T3", "T4", "T5"],
    "price_open": [150, 2800, 300, 700, 600],
    "price_close": [155, 2825, 290, 720, 590],
    "volume": [1000, 500, 2000, 1500, 1200],
    "trade_type": ["buy", "sell", "buy", "buy", "sell"]
})

# Target: 1 if profitable else 0
trade_db['trade_success'] = (trade_db['price_close'] > trade_db['price_open']).astype(int)

# Features and target
X = pd.get_dummies(trade_db[['price_open', 'price_close', 'volume', 'trade_type']], drop_first=True)
y = trade_db['trade_success']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Save model
joblib.dump(clf, "decision_tree_trade_model.joblib")
joblib.dump(X.columns.tolist(), "feature_columns.joblib")  # Save feature columns
print("Model trained and saved successfully!")

#Flask API (app.py)

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model and feature columns
clf = joblib.load("decision_tree_trade_model.joblib")
feature_columns = joblib.load("feature_columns.joblib")

@app.route("/predict", methods=["POST"])
def predict_trade_success():
    """
    Expects JSON payload:
    {
        "price_open": 500,
        "price_close": 520,
        "volume": 1200,
        "trade_type": "buy"
    }
    """
    data = request.json

    # Convert input to DataFrame
    input_df = pd.DataFrame([data])

    # Convert categorical features using get_dummies
    input_df = pd.get_dummies(input_df, columns=['trade_type'], drop_first=True)

    # Align input columns with training columns
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Predict
    prediction = clf.predict(input_df)[0]
    probability = clf.predict_proba(input_df)[0].tolist()

    return jsonify({
        "prediction": int(prediction),
        "probability": probability
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
