# =============================================================================
# CropLogic — CNN Disease Classifier
# Architecture : ResNet-50 fine-tuned on PlantVillage (38 classes)
# Training     : Two-phase protocol (Algorithm 1 in dissertation)
# Dataset      : https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
# =============================================================================
import os, sys, json, pathlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (IMG_SIZE, BATCH_SIZE, PHASE1_EPOCHS, PHASE2_EPOCHS,
                    PHASE1_LR, PHASE2_LR, DROPOUT_RATE, UNFREEZE_LAYERS,
                    DISEASE_CLASSES, NUM_CLASSES, MODEL_DIR, PLANTVILLAGE_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Imports — guarded so the module can be imported without GPU
# ─────────────────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                             ReduceLROnPlateau, CSVLogger)
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[CNN] TensorFlow not available — inference disabled")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Model Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES) -> "keras.Model":
    """
    Build CropLogic CNN:
        ResNet-50 backbone (ImageNet weights)
        → Global Average Pooling
        → Dense(512, ReLU)
        → Dropout(0.5)
        → Dense(num_classes, softmax)
    """
    assert TF_AVAILABLE, "TensorFlow is required to build the model"

    base = ResNet50(
        weights      = "imagenet",
        include_top  = False,
        input_shape  = (*IMG_SIZE, 3),
    )
    base.trainable = False     # frozen in Phase 1

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu", name="fc_head")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    return Model(inputs, outputs, name="CropLogic_ResNet50")


# ─────────────────────────────────────────────────────────────────────────────
# Data Generators
# ─────────────────────────────────────────────────────────────────────────────

def make_generators(data_dir: str):
    """
    Create train / val / test ImageDataGenerators.
    Expects PlantVillage directory structure:
        data_dir/
            train/  <class_name>/*.jpg
            valid/  <class_name>/*.jpg   (or val/)
            test/   <class_name>/*.jpg   (optional)
    """
    train_aug = ImageDataGenerator(
        rescale           = 1.0 / 255,
        horizontal_flip   = True,
        rotation_range    = 15,
        zoom_range        = 0.10,
        brightness_range  = [0.80, 1.20],
        width_shift_range = 0.05,
        height_shift_range= 0.05,
    )
    val_gen = ImageDataGenerator(rescale=1.0 / 255)

    kw = dict(target_size=IMG_SIZE, batch_size=BATCH_SIZE,
              class_mode="categorical", shuffle=True)

    train_dir = os.path.join(data_dir, "train")
    val_dir   = (os.path.join(data_dir, "valid") if
                 os.path.isdir(os.path.join(data_dir, "valid")) else
                 os.path.join(data_dir, "val"))

    train_flow = train_aug.flow_from_directory(train_dir, **kw)
    val_flow   = val_gen.flow_from_directory(val_dir, shuffle=False, **{**kw, "shuffle":False})

    test_flow = None
    test_dir  = os.path.join(data_dir, "test")
    if os.path.isdir(test_dir):
        test_flow = val_gen.flow_from_directory(test_dir, shuffle=False, **{**kw, "shuffle":False})

    return train_flow, val_flow, test_flow


# ─────────────────────────────────────────────────────────────────────────────
# Two-Phase Training  (Algorithm 1)
# ─────────────────────────────────────────────────────────────────────────────

def train(data_dir: str = PLANTVILLAGE_DIR,
          save_dir: str = MODEL_DIR) -> "keras.Model":
    """
    Full two-phase training pipeline.

    Phase 1 — Head training (ResNet-50 frozen):
        Epochs: PHASE1_EPOCHS, LR: PHASE1_LR
    Phase 2 — Fine-tuning (top UNFREEZE_LAYERS unfrozen):
        Epochs: PHASE2_EPOCHS, LR: PHASE2_LR, early stopping patience=5
    """
    assert TF_AVAILABLE, "TensorFlow required for training"
    os.makedirs(save_dir, exist_ok=True)

    train_gen, val_gen, _ = make_generators(data_dir)
    num_classes = train_gen.num_classes
    print(f"[CNN] Classes detected: {num_classes}")
    print(f"[CNN] Train samples  : {train_gen.samples}")
    print(f"[CNN] Val samples    : {val_gen.samples}")

    # Save class index mapping
    with open(os.path.join(save_dir, "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    model = build_model(num_classes)
    model.summary(line_length=100)

    # ── Phase 1: Head only ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1 — Head Training (backbone frozen)")
    print("="*60)
    model.compile(
        optimizer = keras.optimizers.Adam(PHASE1_LR),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )
    callbacks_p1 = [
        ModelCheckpoint(os.path.join(save_dir, "phase1_best.h5"),
                        monitor="val_accuracy", save_best_only=True, verbose=1),
        CSVLogger(os.path.join(save_dir, "phase1_log.csv")),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]
    history1 = model.fit(
        train_gen, epochs=PHASE1_EPOCHS, validation_data=val_gen,
        callbacks=callbacks_p1,
    )

    # ── Phase 2: Fine-tune top N layers ───────────────────────────────────────
    print("\n" + "="*60)
    print(f"PHASE 2 — Fine-Tuning (top {UNFREEZE_LAYERS} layers unfrozen)")
    print("="*60)
    base_model = model.layers[1]   # ResNet50 is layer index 1 after Input
    for layer in base_model.layers[-UNFREEZE_LAYERS:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    model.compile(
        optimizer = keras.optimizers.Adam(PHASE2_LR),
        loss      = "categorical_crossentropy",
        metrics   = ["accuracy", keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")],
    )
    callbacks_p2 = [
        EarlyStopping(monitor="val_accuracy", patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(os.path.join(save_dir, "resnet50_plantvillage.h5"),
                        monitor="val_accuracy", save_best_only=True, verbose=1),
        CSVLogger(os.path.join(save_dir, "phase2_log.csv")),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
    ]
    history2 = model.fit(
        train_gen, epochs=PHASE2_EPOCHS, validation_data=val_gen,
        callbacks=callbacks_p2,
    )

    print(f"\n[CNN] ✓ Model saved to {save_dir}")
    return model, history1, history2


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

class DiseaseClassifier:
    """
    Wraps the trained ResNet-50 model for single-image inference.
    Falls back to a heuristic if no model is available (demo mode).
    """

    def __init__(self, model_path: str = None, class_index_path: str = None):
        self.model         = None
        self.class_names   = DISEASE_CLASSES
        self.idx_to_class  = {i: c for i, c in enumerate(DISEASE_CLASSES)}

        if model_path and os.path.exists(model_path) and TF_AVAILABLE:
            print(f"[CNN] Loading model from {model_path}")
            self.model = keras.models.load_model(model_path)

            # Override class names from saved index if available
            if class_index_path and os.path.exists(class_index_path):
                with open(class_index_path) as f:
                    idx = json.load(f)
                self.idx_to_class = {v: k for k, v in idx.items()}
                self.class_names  = [self.idx_to_class[i] for i in sorted(self.idx_to_class)]
            print(f"[CNN] ✓ Model ready — {len(self.class_names)} classes")
        else:
            print("[CNN] No trained model found — running in heuristic demo mode")

    def preprocess(self, img_path_or_array) -> np.ndarray:
        """Load and preprocess a single image for ResNet-50."""
        assert PIL_AVAILABLE, "Pillow required for image preprocessing"
        if isinstance(img_path_or_array, (str, Path)):
            img = Image.open(img_path_or_array).convert("RGB")
        elif hasattr(img_path_or_array, "read"):   # file-like object
            img = Image.open(img_path_or_array).convert("RGB")
        else:
            img = Image.fromarray(img_path_or_array).convert("RGB")
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(arr, axis=0)   # (1, H, W, 3)

    def predict(self, img_input, top_k: int = 3) -> List[dict]:
        """
        Run inference on a single image.

        Returns list of top-k dicts:
            {"class": str, "confidence": float, "healthy": bool}
        """
        if self.model is None or not TF_AVAILABLE:
            return self._heuristic_predict(top_k)

        arr   = self.preprocess(img_input)
        probs = self.model.predict(arr, verbose=0)[0]   # (38,)
        top_idx = np.argsort(probs)[::-1][:top_k]

        return [
            {
                "class"     : self.idx_to_class.get(i, DISEASE_CLASSES[i]),
                "confidence": round(float(probs[i]), 4),
                "healthy"   : "healthy" in self.idx_to_class.get(i, "").lower(),
            }
            for i in top_idx
        ]

    def _heuristic_predict(self, top_k: int = 3) -> List[dict]:
        """Heuristic fallback when no model is loaded — for UI demos."""
        import random
        classes = random.sample(DISEASE_CLASSES, min(top_k, len(DISEASE_CLASSES)))
        confs   = sorted(np.random.dirichlet(np.ones(top_k) * 0.5).tolist(), reverse=True)
        return [
            {"class": c, "confidence": round(confs[i], 4),
             "healthy": "healthy" in c.lower()}
            for i, c in enumerate(classes)
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, test_gen) -> Dict:
    """Compute Top-1, Top-5, and per-class metrics on the test set."""
    assert TF_AVAILABLE
    loss, top1, top5 = model.evaluate(test_gen, verbose=1)

    # Per-class confusion matrix via sklearn
    try:
        from sklearn.metrics import classification_report, confusion_matrix
        preds_raw = model.predict(test_gen, verbose=1)
        y_pred    = np.argmax(preds_raw, axis=1)
        y_true    = test_gen.classes
        report    = classification_report(y_true, y_pred,
                                           target_names=list(test_gen.class_indices.keys()),
                                           output_dict=True)
        return {"top1": top1, "top5": top5, "loss": loss, "report": report}
    except ImportError:
        return {"top1": top1, "top5": top5, "loss": loss}
