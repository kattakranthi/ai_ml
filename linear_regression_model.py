import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample trade data
data = {
    'quantity': [100, 50, 200, 150, 10, 100, 100, 80, 20, 50],
    'price': [185.5, 190, 99.8, 320, 2050, 100.2, 192, 250, 70, 322],
    'side': [1,0,1,1,1,0,1,1,0,0],   # BUY=1, SELL=0
    'trade_value': [18550, 9500, 19960, 48000, 20500, 10020, 19200, 20000, 1400, 16100]
}

df = pd.DataFrame(data)

# Features and target
X = df[['quantity', 'price', 'side']]
y = df['trade_value']

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Coefficients
print("Weights:", model.coef_)
print("Bias:", model.intercept_)

# Predict
df['predicted_value'] = model.predict(X)
print(df)
