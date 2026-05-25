"""
Model training module for tennis tournament predictor.
Trains and evaluates multiple ML models.
"""
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import time
warnings.filterwarnings('ignore')

# Optional: Intel Extension for Scikit-learn (accelerates SVM, LogisticRegression, etc.)
# Must be called before importing sklearn members
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("[INFO] Intel Extension for Scikit-learn enabled (GPU/CPU Acceleration)")
except ImportError:
    pass

# Scikit-learn models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, log_loss)
from sklearn.inspection import permutation_importance

# XGBoost
import xgboost as xgb

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Configure TensorFlow for GPU usage
# Try to enable memory growth to prevent TF from allocating all VRAM at once
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"[INFO] TensorFlow configured to use GPU: {len(gpus)} device(s) found")
    except RuntimeError as e:
        print(f"[ERROR] TensorFlow GPU configuration failed: {e}")
else:
    print("[INFO] No GPU found for TensorFlow, using CPU.")


class ModelTrainer:
    """Class to train and evaluate multiple ML models."""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.evaluation_results = {}
        self.training_times = {}
        self.feature_names = None
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression model."""
        print("\nTraining Logistic Regression...")
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=1000, random_state=42))
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluation
        results = self._evaluate_model(y_test, y_pred, y_pred_proba, 'Logistic Regression')
        
        # Save model and scaler
        self.models['logistic_regression'] = pipeline
        self.scalers['logistic_regression'] = pipeline.named_steps['scaler']
        
        return results
    
    def train_knn(self, X_train, y_train, X_test, y_test):
        """Train K-Nearest Neighbors model."""
        print("\nTraining KNN...")
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=5))
        ])
        
        # Use smaller sample for KNN if dataset is too large (KNN prediction is very slow)
        if len(X_train) > 50000:
            print(f"  Using sample of 50,000 for KNN training (Dataset size: {len(X_train):,})...")
            sample_idx = np.random.choice(len(X_train), 50000, replace=False)
            X_train_sample = X_train[sample_idx]
            y_train_sample = y_train[sample_idx]
            pipeline.fit(X_train_sample, y_train_sample)
        else:
            pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluation
        results = self._evaluate_model(y_test, y_pred, y_pred_proba, 'KNN')
        
        # Save model and scaler
        self.models['knn'] = pipeline
        self.scalers['knn'] = pipeline.named_steps['scaler']
        
        return results
    
    def train_svm(self, X_train, y_train, X_test, y_test):
        """Train Support Vector Machine model."""
        print("\nTraining SVM...")
        
        # Create pipeline with scaling
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', probability=True, random_state=42))
        ])
        
        # Use smaller sample for SVM if dataset is too large (SVM is slow)
        if len(X_train) > 50000:
            print("  Using sample of 50,000 for SVM training (SVM is computationally expensive)...")
            sample_idx = np.random.choice(len(X_train), 50000, replace=False)
            X_train_sample = X_train[sample_idx]
            y_train_sample = y_train[sample_idx]
            pipeline.fit(X_train_sample, y_train_sample)
        else:
            pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Evaluation
        results = self._evaluate_model(y_test, y_pred, y_pred_proba, 'SVM')
        
        # Save model and scaler
        self.models['svm'] = pipeline
        self.scalers['svm'] = pipeline.named_steps['scaler']
        
        return results
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model."""
        print("\nTraining Random Forest...")
        
        # No scaling needed for Random Forest
        model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluation
        results = self._evaluate_model(y_test, y_pred, y_pred_proba, 'Random Forest')
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        results['feature_importance'] = feature_importance
        
        # Save model
        self.models['random_forest'] = model
        
        return results
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model."""
        print("\nTraining XGBoost...")
        
        # No scaling needed for XGBoost
        # Use GPU for XGBoost if available
        # logic for choosing tree_method based on version
        try:
            # Check for GPU availability in XGBoost
            xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'eval_metric': 'logloss'
            }
            
            # Use 'gpu_hist' or 'hist' with 'device'='cuda'
            # For newer XGBoost versions:
            xgb_params['tree_method'] = 'hist'
            xgb_params['device'] = 'cuda'
            
            print("  Trying XGBoost with GPU acceleration...")
            model = xgb.XGBClassifier(**xgb_params)
            model.fit(X_train, y_train)
        except Exception:
            print("  GPU training failed or not available for XGBoost, falling back to CPU...")
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluation
        results = self._evaluate_model(y_test, y_pred, y_pred_proba, 'XGBoost')
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        results['feature_importance'] = feature_importance
        
        # Save model
        self.models['xgboost'] = model
        
        return results
    
    def train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train Neural Network using TensorFlow/Keras."""
        print("\nTraining Neural Network on CPU...")
        
        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build model
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')]
        )
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        model_checkpoint = callbacks.ModelCheckpoint(
            os.path.join(self.models_dir, 'neural_network_best.h5'),
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train
        history = model.fit(
            X_train_scaled, y_train,
            validation_split=0.25,
            epochs=2500,
            batch_size=1024,
            callbacks=[early_stopping, model_checkpoint],
            verbose=1
        )
        
        # Load best model
        model.load_weights(os.path.join(self.models_dir, 'neural_network_best.h5'))
        
        # Predictions
        y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Evaluation
        results = self._evaluate_model(y_test, y_pred, y_pred_proba, 'Neural Network')
        results['training_history'] = history.history
        
        # Save model and scaler
        model.save(os.path.join(self.models_dir, 'neural_network.h5'))
        self.models['neural_network'] = model
        self.scalers['neural_network'] = scaler
        
        return results
    
    def _evaluate_model(self, y_true, y_pred, y_pred_proba, model_name):
        """Evaluate model performance."""
        results = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'log_loss': log_loss(y_true, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        return results
    
    def train_all_models(self, X_train, y_train, X_test, y_test, feature_names, selected_keys=None):
        """
        Train specific or all models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            feature_names: List of feature names
            selected_keys: List of model keys to train (e.g., ['xgboost', 'random_forest']).
                          If None, trains all models.
        """
        self.feature_names = feature_names
        
        training_map = {
            'logistic_regression': self.train_logistic_regression,
            'knn': self.train_knn,
            'svm': self.train_svm,
            'random_forest': self.train_random_forest,
            'xgboost': self.train_xgboost,
            'neural_network': self.train_neural_network
        }
        
        # If selected_keys is provided, filter the training_map
        if selected_keys:
            keys_to_train = [k for k in selected_keys if k in training_map]
            if not keys_to_train:
                print("\n[WARNING] No valid models selected. Defaulting to all models.")
                keys_to_train = list(training_map.keys())
        else:
            keys_to_train = list(training_map.keys())
            
        print("\n" + "="*60)
        print(f"TRAINING {len(keys_to_train)} MODEL(S)")
        print("="*60)
        
        # Train each selected model
        for key in keys_to_train:
            start_time = time.time()
            self.evaluation_results[key] = training_map[key](X_train, y_train, X_test, y_test)
            end_time = time.time()
            self.training_times[key] = end_time - start_time
            print(f"  [TIME] {key} took {self.training_times[key]:.2f} seconds")
        
        # Save all models (only those that were just trained will be overwritten)
        self.save_models()
        
        # Display results for models trained in this session
        self.display_session_results(keys_to_train)
        
        print("\n" + "="*60)
        print(f"[OK] {len(keys_to_train)} model(s) trained successfully!")
        print("="*60 + "\n")

    def display_session_results(self, session_keys):
        """Display evaluation results for models trained in the current session."""
        print("\n" + "="*60)
        print("SESSION MODEL EVALUATION RESULTS")
        print("="*60)
        
        session_results = {k: self.evaluation_results[k] for k in session_keys if k in self.evaluation_results}
        if not session_results:
            print("No results to display.")
            return

        results_df = pd.DataFrame(session_results).T
        
        # Add training times to the DataFrame
        times = [self.training_times.get(k, 0) for k in session_results.keys()]
        results_df['training_time'] = times
        
        results_df = results_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'log_loss', 'training_time']]
        results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Log Loss', 'Time (s)']
        
        print("\n" + results_df.round(4).to_string())
    
    def display_results(self):
        """Display evaluation results for all models."""
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        results_df = pd.DataFrame(self.evaluation_results).T
        results_df = results_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'log_loss']]
        results_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Log Loss']
        
        print("\n" + results_df.round(4).to_string())
        
        # Best model by each metric
        print("\nBest Models:")
        print(f"  Accuracy: {results_df['Accuracy'].idxmax()} ({results_df['Accuracy'].max():.4f})")
        print(f"  F1-Score: {results_df['F1-Score'].idxmax()} ({results_df['F1-Score'].max():.4f})")
        print(f"  ROC-AUC: {results_df['ROC-AUC'].idxmax()} ({results_df['ROC-AUC'].max():.4f})")
    
    def save_models(self):
        """Save all trained models."""
        print("\nSaving models...")
        
        # Save scikit-learn models
        for name, model in self.models.items():
            if name != 'neural_network':  # Neural network saved separately
                model_path = os.path.join(self.models_dir, f'{name}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                print(f"  [OK] Saved: {model_path}")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(self.models_dir, f'{name}_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"  [OK] Saved: {scaler_path}")
        
        # Save evaluation results
        results_path = os.path.join(self.models_dir, 'evaluation_results.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump(self.evaluation_results, f)
        print(f"  [OK] Saved: {results_path}")
        
        # Save feature names
        feature_names_path = os.path.join(self.models_dir, 'feature_names.pkl')
        with open(feature_names_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        print(f"  [OK] Saved: {feature_names_path}")
    
    def load_models(self):
        """Load previously trained models."""
        print("Loading models...")
        
        # Load scikit-learn models
        model_files = {
            'logistic_regression': 'logistic_regression.pkl',
            'knn': 'knn.pkl',
            'svm': 'svm.pkl',
            'random_forest': 'random_forest.pkl',
            'xgboost': 'xgboost.pkl'
        }
        
        for name, filename in model_files.items():
            model_path = os.path.join(self.models_dir, filename)
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"  [OK] Loaded: {name}")
        
        # Load neural network
        nn_path = os.path.join(self.models_dir, 'neural_network.h5')
        if os.path.exists(nn_path):
            try:
                self.models['neural_network'] = keras.models.load_model(nn_path)
                print(f"  [OK] Loaded: neural_network")
            except (TypeError, ValueError) as e:
                print(f"  [WARN] Could not load neural_network (Keras version mismatch): {e}")
                print(f"         Re-train the neural network (Option 2) to fix this.")
        
        # Load scalers
        scaler_files = {
            'logistic_regression': 'logistic_regression_scaler.pkl',
            'knn': 'knn_scaler.pkl',
            'svm': 'svm_scaler.pkl',
            'neural_network': 'neural_network_scaler.pkl'
        }
        
        for name, filename in scaler_files.items():
            scaler_path = os.path.join(self.models_dir, filename)
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers[name] = pickle.load(f)
        
        # Load evaluation results
        results_path = os.path.join(self.models_dir, 'evaluation_results.pkl')
        if os.path.exists(results_path):
            with open(results_path, 'rb') as f:
                self.evaluation_results = pickle.load(f)
        
        # Load feature names
        feature_names_path = os.path.join(self.models_dir, 'feature_names.pkl')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'rb') as f:
                self.feature_names = pickle.load(f)


if __name__ == "__main__":
    # Test model training
    from data_loader import load_data
    from feature_engineering import prepare_features_for_training, split_data_by_date
    
    print("Testing model training...")
    df = load_data()
    
    # Prepare features (Temporal Symmetric Pass)
    _, _, feature_names, encoders, df_feats = prepare_features_for_training(df)
    
    # Split data temporally
    train_df, test_df = split_data_by_date(df_feats, test_size=0.2)
    
    X_train = train_df[feature_names].values
    y_train = train_df['target'].values
    X_test = test_df[feature_names].values
    y_test = test_df['target'].values
    
    # Train models
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train, X_test, y_test, feature_names)
