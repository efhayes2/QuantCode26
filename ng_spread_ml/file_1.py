import pandas as pd
import ta  # Technical indicators library
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Step 1: Load Historical Data (Replace with your NG futures data)
def load_data():
    # Placeholder: Load March and April NG futures settlement prices
    df = pd.read_csv('natural_gas_futures.csv', parse_dates=['Date'])
    df.sort_values('Date', inplace=True)

    # Compute spread (March - April contract)
    df['Spread'] = df['NG_Mar'] - df['NG_Apr']

    return df


# Step 2: Feature Engineering
def add_features(df):
    df['Spread_Change'] = df['Spread'].diff()
    df['Spread_Roll_Mean'] = df['Spread'].rolling(window=10).mean()
    df['Spread_Roll_Std'] = df['Spread'].rolling(window=10).std()
    df['Spread_ZScore'] = (df['Spread'] - df['Spread_Roll_Mean']) / df['Spread_Roll_Std']
    df['Spread_RSI'] = ta.momentum.rsi(df['Spread'], window=14)
    df['Spread_Boll_Upper'] = ta.volatility.bollinger_hband(df['Spread'], window=20)
    df['Spread_Boll_Lower'] = ta.volatility.bollinger_lband(df['Spread'], window=20)
    df.dropna(inplace=True)  # Remove NaN values
    return df


# Step 3: Prepare Training Data
def prepare_data(df):
    features = ['Spread', 'Spread_Roll_Mean', 'Spread_Roll_Std', 'Spread_ZScore', 'Spread_RSI', 'Spread_Boll_Upper',
                'Spread_Boll_Lower']
    target = 'Spread_Change'  # Predict next-day spread movement

    X = df[features]
    y = df[target].shift(-1)  # Shift target to predict future movement
    df.dropna(inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


# Step 4: Train the Model
def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# Step 5: Evaluate Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'MSE: {mse:.4f}, R²: {r2:.4f}')
    return y_pred


# Step 6: Run Pipeline
df_ = load_data()
df_ = add_features(df_)
X_train_, X_test_, y_train_, y_test_ = prepare_data(df_)
model_ = train_model(X_train_, y_train_)
y_pred_ = evaluate_model(model_, X_test_, y_test_)
