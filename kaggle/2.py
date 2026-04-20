import pandas as pd
import numpy as np
import re
import time
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
import xgboost as xgb
import gensim
from gensim.models import Word2Vec, FastText
from gensim.utils import simple_preprocess
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 1. Загрузка и предобработка
# ------------------------------
print("Загрузка данных...")
TRAIN_PATH = r"C:\Users\annap\VSCode\python\nikolskaya\spbu_ml_2026\kaggle\materials\train.csv"
TEST_PATH = r"C:\Users\annap\VSCode\python\nikolskaya\spbu_ml_2026\kaggle\materials\test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df['full_text'] = train_df['title'].fillna('') + ' ' + train_df['body'].fillna('')
test_df['full_text'] = test_df['title'].fillna('') + ' ' + test_df['body'].fillna('')

# NLTK стоп-слова и лемматизатор
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_and_lemmatize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\'\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 1]
    return ' '.join(words)

def tokenize_for_embeddings(text):
    """Возвращает список слов (токенов) для обучения эмбеддингов"""
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 1]
    return words

print("Очистка текстов...")
train_df['clean_tfidf'] = train_df['full_text'].apply(clean_and_lemmatize)
test_df['clean_tfidf'] = test_df['full_text'].apply(clean_and_lemmatize)

train_tokens = train_df['full_text'].apply(tokenize_for_embeddings).tolist()
test_tokens = test_df['full_text'].apply(tokenize_for_embeddings).tolist()

# ------------------------------
# 2. Расширенные мета-признаки (27)
# ------------------------------
def extract_rich_meta(text):
    if not isinstance(text, str):
        return [0]*27
    lower_text = text.lower()
    chars = len(text)
    words = lower_text.split()
    num_words = len(words)
    num_unique = len(set(words))
    excl = text.count('!')
    ques = text.count('?')
    period = text.count('.')
    comma = text.count(',')
    semicolon = text.count(';')
    colon = text.count(':')
    punct_total = excl + ques + period + comma + semicolon + colon
    caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 1)
    caps_ratio = caps_words / (num_words + 1)
    pos_smile = text.count(':)') + text.count(':D') + text.count(';)') + text.count(':-)')
    neg_smile = text.count(':(') + text.count(":'(") + text.count(':-(')
    newlines = text.count('\n')
    digit_ratio = sum(c.isdigit() for c in text) / (chars + 1)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    depression_markers = [
        'suicide', 'die', 'kill', 'hurt', 'pain', 'alone', 'sad', 'depress',
        'hopeless', 'worthless', 'empty', 'numb', 'cry', 'anxiety', 'panic',
        'tired', 'exhausted', 'hate myself', 'no future', 'end it', 'give up',
        'cut', 'self harm', 'bleed', 'overdose', 'helpless', 'lonely', 'abandon'
    ]
    marker_count = sum(lower_text.count(m) for m in depression_markers)
    marker_density = marker_count / (num_words + 1)
    uniq_ratio = num_unique / (num_words + 1)
    short_words = sum(1 for w in words if len(w) <= 2)
    short_ratio = short_words / (num_words + 1)
    modal_count = sum(lower_text.count(m) for m in ['can', 'could', 'will', 'would', 'should', 'may', 'might'])
    negative_words = ['no', 'not', 'never', 'nothing', 'none', 'nobody', 'neither', 'nor', 'but', 'however', 'unfortunately', 'sadly']
    neg_count = sum(lower_text.count(w) for w in negative_words)
    very_short = 1 if num_words < 3 else 0
    help_phrase = 1 if any(phrase in lower_text for phrase in ['i need help', 'please help', 'can someone help', 'help me']) else 0
    punct_ratio = punct_total / (chars + 1)
    caps_char_ratio = caps_words / (chars + 1)
    return [
        chars, num_words, num_unique, excl, ques, period, comma, semicolon, colon, punct_total,
        caps_words, caps_ratio, pos_smile, neg_smile, newlines, digit_ratio, avg_word_len,
        marker_count, marker_density, uniq_ratio, short_ratio, modal_count, neg_count,
        very_short, help_phrase, punct_ratio, caps_char_ratio
    ]

print("Извлечение мета-признаков...")
train_meta = np.array(train_df['full_text'].apply(extract_rich_meta).tolist())
test_meta = np.array(test_df['full_text'].apply(extract_rich_meta).tolist())

# ------------------------------
# 3. TF-IDF: словные n-граммы (1-3)
# ------------------------------
print("TF-IDF словные n-граммы...")
tfidf_word = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1,3),
    stop_words='english',
    min_df=3,
    max_df=0.90,
    sublinear_tf=True
)
X_train_word = tfidf_word.fit_transform(train_df['clean_tfidf'])
X_test_word = tfidf_word.transform(test_df['clean_tfidf'])

# ------------------------------
# 4. TF-IDF: символьные n-граммы (2-6)
# ------------------------------
print("TF-IDF символьные n-граммы...")
tfidf_char = TfidfVectorizer(
    max_features=20000,
    ngram_range=(2,6),
    analyzer='char_wb',
    min_df=3,
    max_df=0.90,
    sublinear_tf=True
)
X_train_char = tfidf_char.fit_transform(train_df['clean_tfidf'])
X_test_char = tfidf_char.transform(test_df['clean_tfidf'])

# ------------------------------
# 5. Обучение эмбеддингов Word2Vec и FastText
# ------------------------------
print("Обучение Word2Vec (300d)...")
w2v_model = Word2Vec(
    sentences=train_tokens,
    vector_size=300,
    window=7,
    min_count=2,
    workers=4,
    epochs=20,
    sg=1  # skip-gram
)
def get_w2v_embedding(tokens, model, size=300):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(size)
    return np.mean(vectors, axis=0)
train_w2v = np.array([get_w2v_embedding(tok, w2v_model) for tok in train_tokens])
test_w2v = np.array([get_w2v_embedding(tok, w2v_model) for tok in test_tokens])

print("Обучение FastText (300d)...")
ft_model = FastText(
    sentences=train_tokens,
    vector_size=300,
    window=7,
    min_count=2,
    workers=4,
    epochs=20,
    sg=1
)
def get_ft_embedding(tokens, model, size=300):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    if not vectors:
        return np.zeros(size)
    return np.mean(vectors, axis=0)
train_ft = np.array([get_ft_embedding(tok, ft_model) for tok in train_tokens])
test_ft = np.array([get_ft_embedding(tok, ft_model) for tok in test_tokens])

# ------------------------------
# 6. Нормализация всех признаков
# ------------------------------
scaler_meta = StandardScaler()
train_meta_scaled = scaler_meta.fit_transform(train_meta)
test_meta_scaled = scaler_meta.transform(test_meta)

scaler_w2v = StandardScaler()
train_w2v_scaled = scaler_w2v.fit_transform(train_w2v)
test_w2v_scaled = scaler_w2v.transform(test_w2v)

scaler_ft = StandardScaler()
train_ft_scaled = scaler_ft.fit_transform(train_ft)
test_ft_scaled = scaler_ft.transform(test_ft)

# ------------------------------
# 7. Объединение всех признаков в одну разреженную матрицу
# ------------------------------
print("Объединение признаков...")
X_train = hstack([
    X_train_word,
    X_train_char,
    csr_matrix(train_meta_scaled),
    csr_matrix(train_w2v_scaled),
    csr_matrix(train_ft_scaled)
])
X_test = hstack([
    X_test_word,
    X_test_char,
    csr_matrix(test_meta_scaled),
    csr_matrix(test_w2v_scaled),
    csr_matrix(test_ft_scaled)
])
y = train_df['label'].values
print(f"Размер обучающей матрицы: {X_train.shape}")

# ------------------------------
# 8. Стекинг с кросс-валидацией (5 фолдов) для одного seed
# ------------------------------
def stacking_model(seed):
    models = {
        'lgb': lgb.LGBMClassifier(
            n_estimators=800, max_depth=10, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.05,
            scale_pos_weight=(y==0).sum()/(y==1).sum(),
            random_state=seed, n_jobs=-1, verbose=-1
        ),
        'xgb': xgb.XGBClassifier(
            n_estimators=600, max_depth=8, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=0.05,
            scale_pos_weight=(y==0).sum()/(y==1).sum(),
            random_state=seed, n_jobs=-1, use_label_encoder=False, verbosity=0
        ),
        'rf': RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_split=5,
            class_weight='balanced', n_jobs=-1, random_state=seed
        ),
        'et': ExtraTreesClassifier(
            n_estimators=400, max_depth=12, min_samples_split=5,
            class_weight='balanced', n_jobs=-1, random_state=seed
        ),
        'lr': LogisticRegression(
            C=0.5, class_weight='balanced', solver='liblinear', max_iter=1000, random_state=seed
        )
    }
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_preds = {name: np.zeros(len(y)) for name in models}
    test_preds = {name: np.zeros((X_test.shape[0], n_folds)) for name in models}
    
    for name, model in models.items():
        print(f"    Seed {seed}: обучение {name}...")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
            X_tr_fold = X_train[train_idx]
            y_tr_fold = y[train_idx]
            X_val_fold = X_train[val_idx]
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_tr_fold, y_tr_fold)
            if hasattr(model_clone, "predict_proba"):
                oof_preds[name][val_idx] = model_clone.predict_proba(X_val_fold)[:, 1]
                test_preds[name][:, fold] = model_clone.predict_proba(X_test)[:, 1]
            else:
                oof_preds[name][val_idx] = model_clone.predict(X_val_fold)
                test_preds[name][:, fold] = model_clone.predict(X_test)
        test_preds[name] = test_preds[name].mean(axis=1)
    
    X_oof = np.column_stack([oof_preds[name] for name in models])
    X_test_meta = np.column_stack([test_preds[name] for name in models])
    meta_model = LogisticRegression(C=1.0, class_weight='balanced', random_state=seed)
    meta_model.fit(X_oof, y)
    final_probs = meta_model.predict_proba(X_test_meta)[:, 1]
    oof_probs = meta_model.predict_proba(X_oof)[:, 1]
    return final_probs, oof_probs

# ------------------------------
# 9. Ансамбль по трём seed
# ------------------------------
seeds = [42, 123, 999]
all_final_probs = []
all_oof_probs = []
for s in seeds:
    print(f"\n=== Запуск для seed {s} ===")
    fp, op = stacking_model(s)
    all_final_probs.append(fp)
    all_oof_probs.append(op)

final_probs_avg = np.mean(all_final_probs, axis=0)
oof_probs_avg = np.mean(all_oof_probs, axis=0)

# ------------------------------
# 10. Поиск оптимального порога
# ------------------------------
thresholds = np.linspace(0.1, 0.9, 100)
best_f1 = 0
best_thr = 0.5
for thr in thresholds:
    preds = (oof_probs_avg >= thr).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr
print(f"\nОптимальный порог: {best_thr:.3f}, OOF F1 (средний): {best_f1:.4f}")

final_preds = (final_probs_avg >= best_thr).astype(int)

# ------------------------------
# 11. Сохранение результата
# ------------------------------
submission = pd.DataFrame({'id': test_df['id'], 'label': final_preds})
submission.to_csv('submission.csv', index=False)
print("submission.csv сохранён.")

import zipfile
with zipfile.ZipFile('submission.zip', 'w') as zipf:
    zipf.write('submission.csv')
print("submission.zip создан.")