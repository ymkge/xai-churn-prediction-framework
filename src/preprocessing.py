import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """CSVデータの読み込み"""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """データの前処理と特徴量生成"""
    # 欠損値補完（今回はダミーデータなので不要だが、実務を想定して記載）
    df = df.fillna(0)
    
    # 不要なID列の削除
    if 'user_id' in df.columns:
        df = df.drop('user_id', axis=1)
    
    # 特徴量エンジニアリング（例：1ヶ月あたりの進捗率）
    df['progress_per_month'] = df['learning_progress'] / (df['enrollment_months'] + 1)
    
    return df

def split_data(df, target_col='churn', test_size=0.2):
    """学習データとテストデータの分割"""
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = load_data('data/sample_churn_data.csv')
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Data split: Train={X_train.shape}, Test={X_test.shape}")
