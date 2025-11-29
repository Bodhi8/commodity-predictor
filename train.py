"""
Main Training Script for Commodity Price Prediction
Trains multiple models and compares their performance.
"""

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
warnings.filterwarnings('ignore')

from data_loader import CommodityDataLoader
from feature_engineering import FeatureEngineer
from models import get_model, ModelTrainer


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Directional Accuracy
    if len(y_true) > 1:
        actual_direction = np.sign(np.diff(y_true.flatten()))
        pred_direction = np.sign(np.diff(y_pred.flatten()))
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    else:
        directional_accuracy = 0
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy
    }


def plot_predictions(y_true, y_pred, dates=None, title='Predictions vs Actual', save_path=None):
    """Plot actual vs predicted values."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Time series plot
    ax1 = axes[0]
    x_axis = dates if dates is not None else range(len(y_true))
    ax1.plot(x_axis, y_true, label='Actual', color='blue', alpha=0.7)
    ax1.plot(x_axis, y_pred, label='Predicted', color='red', alpha=0.7)
    ax1.set_title(title)
    ax1.set_xlabel('Date' if dates is not None else 'Time Step')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2 = axes[1]
    ax2.scatter(y_true, y_pred, alpha=0.5, s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    ax2.set_xlabel('Actual Price')
    ax2.set_ylabel('Predicted Price')
    ax2.set_title('Predicted vs Actual (Scatter)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()
    return fig


def plot_training_history(train_losses, val_losses, title='Training History', save_path=None):
    """Plot training and validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(train_losses, label='Training Loss', color='blue')
    ax.plot(val_losses, label='Validation Loss', color='red')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Log scale for better visualization
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def plot_model_comparison(results_df, save_path=None):
    """Plot comparison of model performance."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics = ['RMSE', 'MAE', 'MAPE', 'Directional_Accuracy']
    colors = sns.color_palette("husl", len(results_df))
    
    for ax, metric in zip(axes.flatten(), metrics):
        bars = ax.bar(results_df['Model'], results_df[metric], color=colors)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, results_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Model Performance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close()
    return fig


def run_experiment(target_commodity='Crude Oil WTI',
                   start_date='2015-01-01',
                   sequence_length=60,
                   prediction_horizon=1,
                   models_to_train=['lstm', 'bilstm', 'transformer', 'lstm_transformer'],
                   epochs=100,
                   batch_size=32,
                   device='cpu',
                   output_dir='./results'):
    """
    Run full experiment: load data, engineer features, train models, evaluate.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("COMMODITY PRICE PREDICTION EXPERIMENT")
    print("="*70)
    print(f"Target: {target_commodity}")
    print(f"Sequence Length: {sequence_length} days")
    print(f"Prediction Horizon: {prediction_horizon} day(s)")
    print(f"Device: {device}")
    print("="*70)
    
    # Step 1: Load Data
    print("\n[1/5] Loading commodity data...")
    loader = CommodityDataLoader(start_date=start_date)
    raw_data = loader.load_all_data(source='yahoo', verbose=True)
    
    if raw_data is None or target_commodity not in raw_data.columns:
        available = list(raw_data.columns) if raw_data is not None else []
        print(f"Error: Target '{target_commodity}' not found. Available: {available}")
        return None
    
    # Fill missing values
    raw_data = raw_data.ffill().bfill()
    
    # Step 2: Feature Engineering
    print("\n[2/5] Engineering features...")
    fe = FeatureEngineer()
    featured_data = fe.build_features(raw_data, verbose=True)
    
    # Step 3: Prepare Sequences
    print("\n[3/5] Preparing sequences...")
    X_train, y_train, X_val, y_val, X_test, y_test = fe.prepare_sequences(
        featured_data,
        target_col=target_commodity,
        sequence_length=sequence_length,
        horizon=prediction_horizon,
        train_ratio=0.7,
        val_ratio=0.15,
        scale=True
    )
    
    input_size = X_train.shape[2]
    print(f"Input features: {input_size}")
    
    # Step 4: Train Models
    print("\n[4/5] Training models...")
    results = []
    all_predictions = {}
    
    for model_name in models_to_train:
        print(f"\n{'='*50}")
        print(f"Training: {model_name.upper()}")
        print(f"{'='*50}")
        
        # Initialize model
        if model_name == 'transformer':
            model = get_model(model_name, input_size, d_model=64, nhead=4, 
                            num_layers=2, dropout=0.1)
        elif model_name == 'lstm_transformer':
            model = get_model(model_name, input_size, hidden_size=64, d_model=64,
                            nhead=4, dropout=0.2)
        else:
            model = get_model(model_name, input_size, hidden_size=64, 
                            num_layers=2, dropout=0.2)
        
        # Train
        trainer = ModelTrainer(model, device=device, learning_rate=0.001)
        train_losses, val_losses = trainer.train(
            X_train, y_train, X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=15,
            verbose=True
        )
        
        # Plot training history
        plot_training_history(
            train_losses, val_losses,
            title=f'{model_name.upper()} Training History',
            save_path=os.path.join(output_dir, f'{model_name}_training.png')
        )
        
        # Predict on test set
        y_pred_scaled = trainer.predict(X_test)
        y_pred = fe.inverse_transform_target(y_pred_scaled)
        y_actual = fe.inverse_transform_target(y_test)
        
        all_predictions[model_name] = y_pred.flatten()
        
        # Calculate metrics
        metrics = calculate_metrics(y_actual.flatten(), y_pred.flatten())
        metrics['Model'] = model_name.upper()
        results.append(metrics)
        
        print(f"\nTest Results for {model_name.upper()}:")
        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE: {metrics['MAE']:.4f}")
        print(f"  MAPE: {metrics['MAPE']:.2f}%")
        print(f"  R²: {metrics['R2']:.4f}")
        print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.1f}%")
        
        # Plot predictions
        plot_predictions(
            y_actual.flatten(), y_pred.flatten(),
            dates=fe.test_dates if hasattr(fe, 'test_dates') else None,
            title=f'{model_name.upper()}: {target_commodity} Predictions',
            save_path=os.path.join(output_dir, f'{model_name}_predictions.png')
        )
    
    # Step 5: Compare Results
    print("\n[5/5] Comparing models...")
    results_df = pd.DataFrame(results)
    results_df = results_df[['Model', 'RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']]
    
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # Find best model
    best_model = results_df.loc[results_df['RMSE'].idxmin(), 'Model']
    print(f"\nBest Model (by RMSE): {best_model}")
    
    # Plot comparison
    plot_model_comparison(
        results_df,
        save_path=os.path.join(output_dir, 'model_comparison.png')
    )
    
    # Save results
    results_df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
    
    # Create combined predictions plot
    fig, ax = plt.subplots(figsize=(14, 6))
    y_actual = fe.inverse_transform_target(y_test).flatten()
    
    ax.plot(y_actual, label='Actual', color='black', linewidth=2)
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_predictions)))
    for (name, preds), color in zip(all_predictions.items(), colors):
        ax.plot(preds, label=name.upper(), alpha=0.7, color=color)
    
    ax.set_title(f'{target_commodity}: All Model Predictions vs Actual')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nResults saved to: {output_dir}/")
    
    return results_df, all_predictions


if __name__ == "__main__":
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Run experiment
    results, predictions = run_experiment(
        target_commodity='Crude Oil WTI',  # Can change to 'Gold', 'Natural Gas', etc.
        start_date='2015-01-01',
        sequence_length=60,
        prediction_horizon=1,
        models_to_train=['lstm', 'bilstm', 'transformer', 'lstm_transformer'],
        epochs=100,
        batch_size=32,
        device=device,
        output_dir='./results'
    )
