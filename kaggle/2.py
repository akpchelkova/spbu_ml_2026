import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """Очистка, стемминг, удаление стоп-слов"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words and len(w) > 1]
    return ' '.join(words)

def extract_meta_features(text):
    """Извлекает: длина в символах, длина в словах, количество !, ?, смайликов, слов-маркеров"""
    if not isinstance(text, str):
        return [0, 0, 0, 0, 0]
    length_char = len(text)
    words = text.split()
    length_word = len(words)
    num_excl = text.count('!')
    num_ques = text.count('?')
    depression_markers = ['suicide', 'die', 'kill', 'hurt', 'pain', 'alone', 'sad', 
                          'depress', 'hopeless', 'worthless', 'empty', 'numb', 'cry']
    marker_count = sum(text.count(m) for m in depression_markers)
    return [length_char, length_word, num_excl, num_ques, marker_count]

TRAIN_PATH = r"C:\Users\annap\VSCode\python\nikolskaya\spbu_ml_2026\kaggle\materials\train.csv"
TEST_PATH = r"C:\Users\annap\VSCode\python\nikolskaya\spbu_ml_2026\kaggle\materials\test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train: {train_df.shape}, Test: {test_df.shape}")

train_df['full_text'] = train_df['title'].fillna('') + ' ' + train_df['body'].fillna('')
test_df['full_text'] = test_df['title'].fillna('') + ' ' + test_df['body'].fillna('')

train_df['clean_text'] = train_df['full_text'].apply(preprocess_text)
test_df['clean_text'] = test_df['full_text'].apply(preprocess_text)

train_meta = np.array(train_df['full_text'].apply(extract_meta_features).tolist())
test_meta = np.array(test_df['full_text'].apply(extract_meta_features).tolist())

tfidf = TfidfVectorizer(
    max_features=20000,          
    ngram_range=(1, 2),        
    stop_words='english',      
    min_df=2,                 
    max_df=0.95,               
    sublinear_tf=True           
)

X_train_tfidf = tfidf.fit_transform(train_df['clean_text'])
X_test_tfidf = tfidf.transform(test_df['clean_text'])

scaler = StandardScaler(with_mean=False)
train_meta_scaled = scaler.fit_transform(train_meta)
test_meta_scaled = scaler.transform(test_meta)

X_train = hstack([X_train_tfidf, csr_matrix(train_meta_scaled)])
X_test = hstack([X_test_tfidf, csr_matrix(test_meta_scaled)])

y_train = train_df['label']

pos_count = (y_train == 1).sum()
neg_count = (y_train == 0).sum()
scale_pos_weight = neg_count / pos_count

lgb_model = lgb.LGBMClassifier(
    objective='binary',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 9, 11],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0.0, 0.1, 1.0],
    'reg_lambda': [0.0, 0.1, 1.0]
}

random_search = RandomizedSearchCV(
    lgb_model, param_dist, n_iter=15, cv=3, 
    scoring='f1', n_jobs=-1, random_state=42, verbose=1
)

random_search.fit(X_train, y_train)

print("Best params:", random_search.best_params_)
print("Best CV F1:", random_search.best_score_)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

best_model = lgb.LGBMClassifier(**random_search.best_params_, objective='binary', 
                                scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1)
best_model.fit(X_tr, y_tr)

y_val_pred_prob = best_model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_pred_prob >= 0.5).astype(int)
val_f1 = f1_score(y_val, y_val_pred)
print(f"Validation F1: {val_f1:.4f}")

final_model = lgb.LGBMClassifier(**random_search.best_params_, objective='binary',
                                 scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1)
final_model.fit(X_train, y_train)

test_probs = final_model.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= 0.5).astype(int)  

submission = pd.DataFrame({
    'id': test_df['id'],
    'label': test_preds
})
submission.to_csv('submission.csv', index=False)
print("submission.csv создан")