
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

# Попытка импорта продвинутых библиотек с обработкой ошибок
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("XGBoost не установлен, продолжаем без него")
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    print("LightGBM не установлен, продолжаем без него")
    LGB_AVAILABLE = False

# Визуализация
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Matplotlib/Seaborn не установлены, продолжаем без визуализаций")
    PLOTTING_AVAILABLE = False

print("=" * 60)
print("ЗАГРУЗКА И ПРОВЕРКА ДАННЫХ")
print("=" * 60)

def load_and_validate_data():
    """Загрузка и проверка структуры данных"""
    
    # Список необходимых файлов
    required_files = ['train.csv', 'test.csv', 'books.csv', 'users.csv']
    
    # Проверяем существование файлов
    available_files = {}
    for file in required_files:
        if os.path.exists(file):
            available_files[file] = file
        elif os.path.exists(f'../input/{file}'):
            available_files[file] = f'../input/{file}'
        elif os.path.exists(f'/kaggle/input/book-rating-prediction/{file}'):
            available_files[file] = f'/kaggle/input/book-rating-prediction/{file}'
        else:
            print(f"Файл {file} не найден")
    
    # Загружаем доступные файлы
    data = {}
    for file_type, file_path in available_files.items():
        try:
            data[file_type.replace('.csv', '')] = pd.read_csv(file_path)
            print(f"✓ Загружен: {file_type}")
        except Exception as e:
            print(f"✗ Ошибка загрузки {file_type}: {e}")
    
    return data

def create_demo_data():
    """Создание демо-данных если файлы не найдены"""
    print("Создание демо-данных...")
    
    np.random.seed(42)
    n_users = 1000
    n_books = 500
    
    # Train data
    train_data = []
    for i in range(5000):
        user_id = np.random.randint(1, n_users + 1)
        book_id = np.random.randint(1, n_books + 1)
        has_read = np.random.choice([0, 1], p=[0.3, 0.7])
        rating = np.random.uniform(5, 10) if has_read == 1 else 0
        year = np.random.randint(2018, 2024)
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)
        timestamp = f"{year}-{month:02d}-{day:02d} {np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}:{np.random.randint(0,60):02d}"
        
        train_data.append({
            'user_id': user_id,
            'book_id': book_id,
            'has_read': has_read,
            'rating': round(rating, 1),
            'timestamp': timestamp
        })
    
    train = pd.DataFrame(train_data)
    
    # Test data (без timestamp и rating)
    test_data = []
    test_users = np.random.choice(range(1, n_users + 1), 200, replace=False)
    for user_id in test_users:
        book_id = np.random.randint(1, n_books + 1)
        test_data.append({
            'user_id': user_id,
            'book_id': book_id
        })
    
    test = pd.DataFrame(test_data)
    
    # Books data
    books_data = []
    for i in range(1, n_books + 1):
        books_data.append({
            'book_id': i,
            'title': f'Book Title {i}',
            'author_id': np.random.randint(1, 101),
            'author_name': f'Author {np.random.randint(1, 101)}',
            'publication_year': np.random.randint(1950, 2023),
            'language': np.random.randint(1, 4),
            'avg_rating': round(np.random.uniform(6, 9), 2),
            'publisher': np.random.randint(1, 11)
        })
    
    books = pd.DataFrame(books_data)
    
    # Users data
    users_data = []
    for i in range(1, n_users + 1):
        users_data.append({
            'user_id': i,
            'gender': np.random.randint(1, 3),
            'age': np.random.randint(18, 70)
        })
    
    users = pd.DataFrame(users_data)
    
    # Genres data
    genres = pd.DataFrame({
        'genre_id': range(1, 11),
        'genre_name': [f'Genre_{i}' for i in range(1, 11)],
        'books_count': [np.random.randint(50, 200) for _ in range(10)]
    })
    
    # Book genres data
    book_genres_data = []
    for book_id in range(1, n_books + 1):
        n_genres = np.random.randint(1, 4)
        genres_for_book = np.random.choice(range(1, 11), n_genres, replace=False)
        for genre_id in genres_for_book:
            book_genres_data.append({
                'book_id': book_id,
                'genre_id': genre_id
            })
    
    book_genres = pd.DataFrame(book_genres_data)
    
    # Book descriptions data
    book_descriptions_data = []
    descriptions = [
        "Интересная книга о приключениях и путешествиях",
        "Роман о любви и отношениях в современном мире", 
        "Фантастическое произведение о будущем человечества",
        "Детективная история с неожиданной развязкой",
        "Исторический роман о важных событиях прошлого",
        "Научная литература о последних открытиях",
        "Биография известного человека",
        "Поэтический сборник современных авторов",
        "Учебное пособие для студентов",
        "Книга о саморазвитии и личностном росте"
    ]
    
    for book_id in range(1, n_books + 1):
        book_descriptions_data.append({
            'book_id': book_id,
            'description': np.random.choice(descriptions)
        })
    
    book_descriptions = pd.DataFrame(book_descriptions_data)
    
    return {
        'train': train,
        'test': test, 
        'books': books,
        'users': users,
        'genres': genres,
        'book_genres': book_genres,
        'book_descriptions': book_descriptions
    }

# Загрузка данных
data = load_and_validate_data()

# Если не все файлы загружены, создаем демо-данные
if len(data) < 4:  # Минимум train, test, books, users
    print("Недостаточно данных, создаем демо-данные...")
    data = create_demo_data()

# Извлекаем данные
train = data['train']
test = data['test']
books = data['books']
users = data['users']

# Дополнительные файлы (могут отсутствовать)
genres = data.get('genres', pd.DataFrame())
book_genres = data.get('book_genres', pd.DataFrame())
book_descriptions = data.get('book_descriptions', pd.DataFrame())

print(f"\nСтруктура данных:")
print(f"  Train: {train.shape}, колонки: {list(train.columns)}")
print(f"  Test: {test.shape}, колонки: {list(test.columns)}")
print(f"  Books: {books.shape}, колонки: {list(books.columns)}")
print(f"  Users: {users.shape}, колонки: {list(users.columns)}")

# ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА
print("\n" + "=" * 60)
print("ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ")
print("=" * 60)

# Обработка временных меток (только если есть в train)
if 'timestamp' in train.columns:
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    print("✓ Временные метки обработаны в train")
else:
    print("✗ Колонка timestamp отсутствует в train")

# В test обычно нет timestamp, поэтому пропускаем
if 'timestamp' in test.columns:
    test['timestamp'] = pd.to_datetime(test['timestamp'])
    print("✓ Временные метки обработаны в test")
else:
    print("✓ В test нет timestamp (ожидаемое поведение)")

# Фильтруем только прочитанные книги для обучения
if 'has_read' in train.columns:
    train_read = train[train['has_read'] == 1].copy()
    print(f"✓ Прочитанных книг в train: {len(train_read)}")
else:
    # Если нет has_read, считаем все книги прочитанными
    train_read = train.copy()
    print("✓ Колонка has_read отсутствует, используем все записи")

# БАЗОВЫЕ ПРИЗНАКИ
print("\nСоздание базовых признаков...")

# Пользовательские статистики (только по прочитанным книгам)
user_stats = train_read.groupby('user_id').agg({
    'rating': ['mean', 'std', 'count', 'min', 'max'],
    'book_id': 'nunique'
}).reset_index()

user_stats.columns = ['user_id', 'user_rating_mean', 'user_rating_std', 
                     'user_books_count', 'user_rating_min', 'user_rating_max', 'user_unique_books']

# Заполняем пропуски в std
user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)

print(f"✓ Создано пользовательских статистик: {len(user_stats)}")

# Книжные статистики
book_stats = train_read.groupby('book_id').agg({
    'rating': ['mean', 'std', 'count'],
    'user_id': 'nunique'
}).reset_index()

book_stats.columns = ['book_id', 'book_rating_mean', 'book_rating_std', 
                     'book_rating_count', 'book_unique_users']

# Заполняем пропуски
book_stats['book_rating_std'] = book_stats['book_rating_std'].fillna(0)

print(f"✓ Создано книжных статистик: {len(book_stats)}")

# Объединение базовых признаков книг
books_features = books.merge(book_stats, on='book_id', how='left')

# Добавляем жанровые признаки если есть
if not book_genres.empty:
    genre_counts = book_genres.groupby('book_id').size().reset_index(name='genre_count')
    books_features = books_features.merge(genre_counts, on='book_id', how='left')
    print("✓ Добавлены жанровые признаки")
else:
    books_features['genre_count'] = 1  # Значение по умолчанию
    print("✓ Жанровые признаки отсутствуют, используем значение по умолчанию")

# Заполнение пропусков в книжных признаках
books_features['book_rating_mean'] = books_features['book_rating_mean'].fillna(books_features.get('avg_rating', 7.0))
books_features['book_rating_std'] = books_features['book_rating_std'].fillna(0)
books_features['book_rating_count'] = books_features['book_rating_count'].fillna(0)
books_features['book_unique_users'] = books_features['book_unique_users'].fillna(0)
books_features['genre_count'] = books_features['genre_count'].fillna(1)

print("✓ Заполнены пропуски в книжных признаках")

# Кодирование категориальных признаков
label_encoders = {}
categorical_columns = ['author_id', 'language', 'publisher']

for col in categorical_columns:
    if col in books_features.columns:
        try:
            le = LabelEncoder()
            books_features[col] = le.fit_transform(books_features[col].astype(str))
            label_encoders[col] = le
            print(f"✓ Закодирована колонка: {col}")
        except Exception as e:
            print(f"✗ Ошибка кодирования {col}: {e}")
            # Если ошибка, создаем числовой признак
            books_features[col] = range(len(books_features))

# Объединение пользовательских признаков
user_features_final = user_stats.merge(users, on='user_id', how='left')

# Заполняем пропуски в пользовательских признаках
if 'gender' in user_features_final.columns:
    user_features_final['gender'] = user_features_final['gender'].fillna(user_features_final['gender'].mode()[0] if len(user_features_final) > 0 else 1)
if 'age' in user_features_final.columns:
    user_features_final['age'] = user_features_final['age'].fillna(user_features_final['age'].median() if len(user_features_final) > 0 else 30)

print("✓ Созданы пользовательские признаки")

# TF-IDF ПРИЗНАКИ
print("\nСоздание TF-IDF признаков...")

def create_tfidf_features(book_descriptions, max_features=100, n_components=20):
    """Создание TF-IDF признаков из описаний книг"""
    
    if book_descriptions.empty or 'description' not in book_descriptions.columns:
        print("✗ Описания книг недоступны, пропускаем TF-IDF")
        return pd.DataFrame({'book_id': books_features['book_id']}), None, None
    
    # Заполнение пропущенных описаний
    book_descriptions['description'] = book_descriptions['description'].fillna('нет описания')
    
    try:
        # Создание TF-IDF векторизатора
        tfidf = TfidfVectorizer(
            max_features=max_features,
            stop_words=['и', 'в', 'на', 'с', 'по', 'о', 'это', 'как', 'для'],
            ngram_range=(1, 2),
            min_df=1
        )
        
        # Обучение TF-IDF и преобразование описаний
        tfidf_features = tfidf.fit_transform(book_descriptions['description'])
        
        # Уменьшение размерности с помощью SVD
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        tfidf_reduced = svd.fit_transform(tfidf_features)
        
        # Создание DataFrame с TF-IDF признаками
        tfidf_columns = [f'tfidf_{i}' for i in range(tfidf_reduced.shape[1])]
        tfidf_df = pd.DataFrame(tfidf_reduced, columns=tfidf_columns)
        tfidf_df['book_id'] = book_descriptions['book_id'].values
        
        print(f"✓ Создано TF-IDF признаков: {tfidf_reduced.shape[1]}")
        return tfidf_df, tfidf, svd
        
    except Exception as e:
        print(f"✗ Ошибка при создании TF-IDF признаков: {e}")
        return pd.DataFrame({'book_id': book_descriptions['book_id']}), None, None

# Создание TF-IDF признаков
tfidf_df, tfidf_vectorizer, svd_transformer = create_tfidf_features(book_descriptions)

# Объединение TF-IDF признаков с основными признаками книг
books_features = books_features.merge(tfidf_df, on='book_id', how='left')

# Заполнение пропусков в TF-IDF признаках
tfidf_columns = [col for col in books_features.columns if col.startswith('tfidf_')]
for col in tfidf_columns:
    books_features[col] = books_features[col].fillna(0)

# СОЗДАНИЕ ФИНАЛЬНОГО ДАТАСЕТА
print("\n" + "=" * 60)
print("СОЗДАНИЕ ФИНАЛЬНОГО ДАТАСЕТА")
print("=" * 60)

# Для обучения
train_final = train_read.merge(user_features_final, on='user_id', how='left')
train_final = train_final.merge(books_features, on='book_id', how='left')

# Определение всех признаков
base_features = [
    'user_rating_mean', 'user_rating_std', 'user_books_count', 
    'user_rating_min', 'user_rating_max', 'user_unique_books'
]

# Добавляем пользовательские демографические признаки если есть
if 'gender' in user_features_final.columns:
    base_features.append('gender')
if 'age' in user_features_final.columns:
    base_features.append('age')

# Книжные признаки
book_feature_candidates = [
    'book_rating_mean', 'book_rating_std', 'book_rating_count', 'book_unique_users',
    'publication_year', 'language', 'publisher', 'avg_rating', 'genre_count', 'author_id'
]

for feat in book_feature_candidates:
    if feat in books_features.columns:
        base_features.append(feat)

# TF-IDF признаки
tfidf_features = [col for col in books_features.columns if col.startswith('tfidf_')]

all_features = base_features + tfidf_features

print(f"Всего признаков: {len(all_features)}")
print(f"  Базовые: {len(base_features)}")
print(f"  TF-IDF: {len(tfidf_features)}")

# Проверяем наличие всех признаков
missing_features = [feat for feat in all_features if feat not in train_final.columns]
if missing_features:
    print(f"✗ Отсутствующие признаки: {missing_features}")
    # Удаляем отсутствующие признаки из списка
    all_features = [feat for feat in all_features if feat in train_final.columns]

print(f"✓ Используется признаков: {len(all_features)}")

# Удаление строк с пропущенными значениями в базовых признаках
initial_size = len(train_final)
train_final = train_final.dropna(subset=base_features)
print(f"✓ Удалено строк с пропущенными значениями: {initial_size - len(train_final)}")

print(f"✓ Размер финального train датасета: {train_final.shape}")

if len(train_final) == 0:
    raise ValueError("Нет данных для обучения после очистки!")

X = train_final[all_features]
y = train_final['rating']

# Подготовка тестовых данных
test_final = test.merge(user_features_final, on='user_id', how='left')
test_final = test_final.merge(books_features, on='book_id', how='left')

# Заполнение пропусков в тестовых данных
for col in all_features:
    if col in test_final.columns and test_final[col].isna().any():
        if col in X.columns:
            test_final[col] = test_final[col].fillna(X[col].median())
        else:
            test_final[col] = test_final[col].fillna(0)

X_test = test_final[all_features]

print(f"✓ Размер тестовых данных: {X_test.shape}")

# ОБУЧЕНИЕ МОДЕЛЕЙ
print("\n" + "=" * 60)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("=" * 60)

# Разделение данных для валидации
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Размеры данных: Train {X_train.shape}, Val {X_val.shape}")

# Модели
models = {}

# 1. Random Forest
print("\n1. Обучение Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model

# 2. Gradient Boosting
print("2. Обучение Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)
models['Gradient Boosting'] = gb_model

# 3. XGBoost (если доступен)
if XGB_AVAILABLE:
    print("3. Обучение XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model

# 4. Linear Models
print("4. Обучение линейных моделей...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
models['Linear Regression'] = lr_model

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
models['Ridge'] = ridge_model

# ОЦЕНКА МОДЕЛЕЙ
print("\n" + "=" * 60)
print("ОЦЕНКА МОДЕЛЕЙ")
print("=" * 60)

def evaluate_model(model, X_val, y_val, model_name):
    """Оценка модели на валидационной выборке"""
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    
    print(f"{model_name}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {np.sqrt(mse):.4f}")
    
    return mse, mae, y_pred

model_performance = {}

for name, model in models.items():
    print(f"\n{name}:")
    mse, mae, y_pred = evaluate_model(model, X_val, y_val, name)
    model_performance[name] = {'mse': mse, 'mae': mae, 'predictions': y_pred}

# АНСАМБЛИРОВАНИЕ
print("\n" + "=" * 60)
print("АНСАМБЛИРОВАНИЕ МОДЕЛЕЙ")
print("=" * 60)

# Простое усреднение
print("Создание ансамбля усреднением...")

class SimpleEnsemble:
    def __init__(self, models):
        self.models = models
        
    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for model in self.models.values():
            predictions += model.predict(X)
        return predictions / len(self.models)

# Создаем ансамбль из лучших моделей
best_model_names = ['Random Forest', 'Gradient Boosting']
if XGB_AVAILABLE:
    best_model_names.append('XGBoost')

ensemble_models = {name: models[name] for name in best_model_names if name in models}

simple_ensemble = SimpleEnsemble(ensemble_models)
ensemble_pred = simple_ensemble.predict(X_val)
ensemble_mse = mean_squared_error(y_val, ensemble_pred)
ensemble_mae = mean_absolute_error(y_val, ensemble_pred)

print(f"Ансамбль ({', '.join(best_model_names)}):")
print(f"  MSE: {ensemble_mse:.4f}")
print(f"  MAE: {ensemble_mae:.4f}")

# ВЫБОР ЛУЧШЕЙ МОДЕЛИ
print("\n" + "=" * 60)
print("ВЫБОР ЛУЧШЕЙ МОДЕЛИ")
print("=" * 60)

# Сравнение всех подходов
final_candidates = {
    'Random Forest': model_performance['Random Forest']['mse'],
    'Gradient Boosting': model_performance['Gradient Boosting']['mse'],
    'Simple Ensemble': ensemble_mse
}

if 'XGBoost' in model_performance:
    final_candidates['XGBoost'] = model_performance['XGBoost']['mse']

best_model_name = min(final_candidates, key=final_candidates.get)
print(f"Лучшая модель: {best_model_name} (MSE: {final_candidates[best_model_name]:.4f})")

# ФИНАЛЬНЫЕ ПРЕДСКАЗАНИЯ
print("\n" + "=" * 60)
print("ФИНАЛЬНЫЕ ПРЕДСКАЗАНИЯ")
print("=" * 60)

# Выбор финальной модели для предсказаний
if best_model_name == 'Simple Ensemble':
    final_model = simple_ensemble
else:
    final_model = models[best_model_name]

# Предсказания на тестовых данных
print(f"Создание предсказаний с помощью {best_model_name}...")
test_predictions = final_model.predict(X_test)

# Ограничение предсказаний диапазоном [0, 10]
test_predictions = np.clip(test_predictions, 0, 10)

# Создание файла решения
submission = test[['user_id', 'book_id']].copy()
submission['rating_predict'] = test_predictions

# Сохранение результатов
submission_file = 'final_submission.csv'
submission.to_csv(submission_file, index=False)

print(f"✓ Файл решения сохранен как: {submission_file}")
print(f"✓ Размер файла решения: {submission.shape}")
print(f"✓ Статистика предсказаний:")
print(f"    Минимальное значение: {submission['rating_predict'].min():.4f}")
print(f"    Максимальное значение: {submission['rating_predict'].max():.4f}")
print(f"    Среднее значение: {submission['rating_predict'].mean():.4f}")
print(f"    Стандартное отклонение: {submission['rating_predict'].std():.4f}")

print("\n" + "=" * 60)
print("АНАЛИЗ ЗАВЕРШЕН!")
print("=" * 60)
print("Итоговые результаты:")
for name, perf in model_performance.items():
    print(f"  {name}: MSE = {perf['mse']:.4f}")

print(f"  Simple Ensemble: MSE = {ensemble_mse:.4f}")

print(f"\nЛучшая модель: {best_model_name}")
print(f"Файл для submission: {submission_file}")
print("=" * 60)