import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Загрузка данных
data = pd.read_csv('enhanced_anxiety_dataset.csv')

# Определение признаков и целевой переменной
X = data.drop('Anxiety Level (1-10)', axis=1)  # Признаки
y = data['Anxiety Level (1-10)']  # Целевая переменная

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Определение категориальных и числовых признаков
categorical_features = ['Gender', 'Occupation', 'Smoking', 'Family History of Anxiety', 
                       'Dizziness', 'Medication', 'Recent Major Life Event']
numeric_features = ['Age', 'Sleep Hours', 'Physical Activity (hrs/week)', 
                    'Caffeine Intake (mg/day)', 'Alcohol Consumption (drinks/week)',
                    'Stress Level (1-10)', 'Heart Rate (bpm)', 'Breathing Rate (breaths/min)',
                    'Sweating Level (1-5)', 'Therapy Sessions (per month)', 'Diet Quality (1-10)']

# Создание преобразователя для категориальных признаков
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Создание пайплайна с предобработкой и моделью
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Обучение модели
model.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")
