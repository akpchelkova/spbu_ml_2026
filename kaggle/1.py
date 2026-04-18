import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('C:/Users/annap/VSCode/python/nikolskaya/spbu_ml_2026/kaggle/period_1_train_data.csv')
test = pd.read_csv('C:/Users/annap/VSCode/python/nikolskaya/spbu_ml_2026/kaggle/test_x.csv')
print("Train shape:", train.shape)
print("Test shape:", test.shape)

train.replace(-999.0, np.nan, inplace=True)
test.replace(-999.0, np.nan, inplace=True)

def fix_rooms(x):
    if x == 'студия':
        return 0
    try:
        return float(x)
    except:
        return np.nan

train['rooms_4'] = train['rooms_4'].apply(fix_rooms)
test['rooms_4'] = test['rooms_4'].apply(fix_rooms)

for df in [train, test]:
    df['agreement_date'] = pd.to_datetime(df['agreement_date'])
    df['year'] = df['agreement_date'].dt.year
    df['month'] = df['agreement_date'].dt.month
    df['day'] = df['agreement_date'].dt.day
    df['dayofweek'] = df['agreement_date'].dt.dayofweek
    df.drop('agreement_date', axis=1, inplace=True)

target = 'price_target'
X_train = train.drop(target, axis=1)
y_train = train[target]
X_test = test.drop('id', axis=1)

cat_cols = ['region_name_cat', 'district_cat', 'corpus_cat', 'developer_cat',
            'hc_name_cat', 'interior_cat', 'class_cat', 'stage_cat']

num_cols = [c for c in X_train.columns if c not in cat_cols]

print(f"Числовых признаков: {len(num_cols)}")
print(f"Категориальных признаков: {len(cat_cols)}")
print(f"Пример числовых: {num_cols[:5]}")
print(f"Пример категориальных: {cat_cols}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

models = {
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'ElasticNet': ElasticNet(random_state=42, max_iter=10000),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0, n_jobs=-1),
    'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=-1)
}

results = {}
print("\nОценка моделей на кросс-валидации (5-fold):")
print("-" * 60)

for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    try:
        scores = cross_val_score(pipeline, X_train, y_train,
                                 cv=5, scoring='neg_mean_absolute_percentage_error',
                                 n_jobs=-1)
        mape = -scores.mean()
        results[name] = mape
        print(f"{name:15} | MAPE = {mape:.4f} (+/- {scores.std():.4f})")
    except Exception as e:
        print(f"{name:15} | Ошибка: {str(e)[:80]}")

if results:
    best_model_name = min(results, key=results.get)
    print(f"\n Лучшая модель: {best_model_name} с MAPE = {results[best_model_name]:.4f}")
    
    print("\nПодбор параметров для RandomForest...")
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }
    pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))])
    grid = GridSearchCV(pipeline_rf, param_grid, cv=3, 
                        scoring='neg_mean_absolute_percentage_error', 
                        n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Лучший RandomForest MAPE: {-grid.best_score_:.4f}")
    print(f"Лучшие параметры: {grid.best_params_}")
    
    print(f"\nОбучение финальной модели ({best_model_name})...")
    best_model = models[best_model_name]
    final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('regressor', best_model)])
    final_pipeline.fit(X_train, y_train)
    
    print("Предсказание на тестовых данных...")
    y_pred = final_pipeline.predict(X_test)
  
    submission = pd.DataFrame({
        'id': test['id'],
        'price_target': y_pred
    })
    submission.to_csv('submission.csv', index=False)
    print("\n Submission сохранён в submission.csv")
    print(f" Количество предсказаний: {len(submission)}")
    print(f" Диапазон предсказаний: {y_pred.min():.2f} - {y_pred.max():.2f}")
    print(f" Среднее предсказание: {y_pred.mean():.2f}")
else:
    print("Не удалось обучить ни одну модель!")