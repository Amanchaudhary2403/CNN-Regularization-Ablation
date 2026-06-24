import os
import json
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

import Config
from models import build_model,VARIENTS
import data

np.random.seed(Config.seed)
tf.random.set_seed(Config.seed)

def run_experiment():
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data.load_data()

    results = {}
    histories = {}

    tf.keras.backend.clear_session()

    for name in VARIENTS.keys():
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        model = build_model(name)

        total_steps = (Config.train_size // Config.batch_size) * Config.epochs
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=Config.lr,
            decay_steps=total_steps,
            alpha=1e-6
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)

        model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=Config.label_smoothing),
            metrics=['accuracy']
        )

        callback = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
                min_delta=1e-4,
                verbose=1,
            ),
            # FIX: added ModelCheckpoint so the best weights survive even if
            # EarlyStopping restore_best_weights fails (known Keras edge-case)
            keras.callbacks.ModelCheckpoint(
                filepath=f"{Config.RESULTS_DIR}/{name}.weights.h5",
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            ),
        ]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

        train_pipeline = (train_dataset
                  .shuffle(buffer_size=len(x_train)) # Shuffle your 5000 images
                  .batch(Config.batch_size)          # Combines images into shape (32, 32, 32, 3)
                  .cache()
                  .prefetch(buffer_size=tf.data.AUTOTUNE))

        val_pipeline = (val_dataset
                .batch(Config.batch_size)            # Batch your validation data too!
                .cache()
                .prefetch(buffer_size=tf.data.AUTOTUNE))

        history = model.fit(train_pipeline,epochs=Config.epochs,callbacks=callback,validation_data=val_pipeline,batch_size=Config.batch_size,verbose=2)

        histories[name] = history.history

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        
        final_train_acc = history.history["accuracy"][-1]
        final_val_acc   = max(history.history["val_accuracy"])
        overfit_gap     = final_train_acc - final_val_acc
 
        results[name] = {
            "test_loss":        round(float(test_loss), 4),
            "test_accuracy":    round(float(test_acc),  4),
            "epochs_trained":   len(history.history["loss"]),
            "best_val_accuracy": round(float(final_val_acc), 4),
            "final_train_accuracy": round(float(final_train_acc), 4),
            "overfit_gap":      round(float(overfit_gap), 4),
        }
 
        print(f"  → Test Acc: {test_acc:.4f}  |  Overfit gap: {overfit_gap:.4f}")
 

        # Save weights for GRAD-CAM
        model.save_weights(f'{Config.RESULTS_DIR}/{name}.weights.h5')
        print(f"Test Accuracy: {test_acc:.4f}")

    # Save results
    with open(f'{Config.RESULTS_DIR}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_results(histories, results)
    print(f"\nDone. Results saved to {Config.RESULTS_DIR}/")

# ═════════════════════════════════════════════════════════════════════════════
# Plotting
# ═════════════════════════════════════════════════════════════════════════════
 
COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
 
 
def plot_results(histories: dict, results: dict):
    plt.style.use("seaborn-v0_8-whitegrid")
 
    _plot_val_accuracy(histories)
    _plot_test_accuracy_bars(results)
    _plot_generalization_gap(histories)
    _plot_loss_grid(histories)
    _plot_overfit_summary(results)
 
    print(f"  Figures saved to '{Config.FIGURES_DIR}/'")
 
 
def _plot_val_accuracy(histories):
    plt.figure(figsize=(11, 6))
    for i, (name, hist) in enumerate(histories.items()):
        plt.plot(hist["val_accuracy"], label=name,
                 color=COLORS[i % len(COLORS)], linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Validation Accuracy", fontsize=12)
    plt.title("Validation Accuracy Over Training", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{Config.FIGURES_DIR}/val_accuracy_curves.png", dpi=300)
    plt.close()
 
 
def _plot_test_accuracy_bars(results):
    names = list(results.keys())
    accs  = [results[n]["test_accuracy"] for n in names]
 
    plt.figure(figsize=(12, 6))
    bars = plt.bar(names, accs, color=COLORS[: len(names)],
                   edgecolor="black", linewidth=1.1)
    plt.ylim(0, 1.0)
    plt.ylabel("Test Accuracy", fontsize=12)
    plt.title("Final Test Accuracy by Variant", fontsize=14, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{acc:.3f}",
            ha="center", fontsize=10, fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(f"{Config.FIGURES_DIR}/test_accuracy_bars.png", dpi=300)
    plt.close()
 
 
def _plot_generalization_gap(histories):
    """
    Plots train_acc − val_acc per epoch.
    FIX: original plotted train_loss − val_loss and labelled it
    'Negative = Overfitting' — which is backwards (higher loss gap
    means train loss < val loss, i.e. the model generalises poorly).
    Accuracy gap is cleaner: positive means train > val (overfit).
    """
    plt.figure(figsize=(11, 6))
    for i, (name, hist) in enumerate(histories.items()):
        gap = np.array(hist["accuracy"]) - np.array(hist["val_accuracy"])
        plt.plot(gap, label=name, color=COLORS[i % len(COLORS)], linewidth=2)
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Train Acc − Val Acc", fontsize=12)
    plt.title("Generalisation Gap (positive = overfit)", fontsize=14, fontweight="bold")
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{Config.FIGURES_DIR}/generalization_gap.png", dpi=300)
    plt.close()
 
 
def _plot_loss_grid(histories):
    """Per-variant train vs val loss subplot grid."""
    n     = len(histories)
    ncols = 3
    nrows = (n + ncols - 1) // ncols     # FIX: dynamic grid size (was hard-coded 2×4)
 
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()
 
    for idx, (name, hist) in enumerate(histories.items()):
        ax = axes[idx]
        ep = range(1, len(hist["loss"]) + 1)
        ax.plot(ep, hist["loss"],     label="Train", linewidth=2)
        ax.plot(ep, hist["val_loss"], label="Val",   linewidth=2, linestyle="--")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss",  fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
 
    for idx in range(n, len(axes)):
        axes[idx].axis("off")
 
    plt.suptitle("Train vs Validation Loss — Per Variant",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{Config.FIGURES_DIR}/train_val_loss_grid.png", dpi=300)
    plt.close()
 
 
def _plot_overfit_summary(results):
    """
    NEW: bar chart of overfit gap (train_acc − val_acc at final epoch).
    Directly answers: 'which variant generalises best?'
    """
    names = list(results.keys())
    gaps  = [results[n]["overfit_gap"] for n in names]
 
    plt.figure(figsize=(12, 5))
    bars = plt.bar(names, gaps, color=COLORS[: len(names)],
                   edgecolor="black", linewidth=1.1)
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.ylabel("Train Acc − Val Acc (final epoch)", fontsize=12)
    plt.title("Overfit Gap by Variant  (lower = better generalisation)",
              fontsize=13, fontweight="bold")
    plt.xticks(rotation=25, ha="right")
    for bar, gap in zip(bars, gaps):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{gap:.3f}",
            ha="center", fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(f"{Config.FIGURES_DIR}/overfit_gap_bars.png", dpi=300)
    plt.close()
 
 
# ═════════════════════════════════════════════════════════════════════════════
 
if __name__ == "__main__":
    run_experiment()
 