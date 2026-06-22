import tensorflow as tf
import keras
from keras.applications import EfficientNetB0
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
import numpy as np
import cv2

class NutritionModel:
    def __init__(self, num_classes=101, weights_path=None):
        self.num_classes = num_classes
        self.model = self._build_model()
        if weights_path:
            try:
                # Use by_name and skip_mismatch to handle potential naming differences in layers
                self.model.load_weights(weights_path, by_name=True, skip_mismatch=True)
                print(f"Weights loaded successfully from {weights_path}")
            except Exception as e:
                print(f"Warning: Could not load weights perfectly: {e}")
        
    def _build_model(self):
        """
        Builds a Transfer Learning model using EfficientNetB0 as the base.
        The base layers are frozen to leverage pre-trained ImageNet weights.
        """
        # Load EfficientNetB0 pre-trained on ImageNet without the top classification layer
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # Freeze the backbone to keep pre-trained feature extraction patterns
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x) # Regularization
        predictions = Dense(self.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def preprocess_image(self, image):
        """
        Resize and normalize image for EfficientNetB0.
        Handles conversion from RGBA to RGB if necessary.
        Input: PIL Image or Numpy array
        Output: Preprocessed tensor
        """
        # Ensure 3 channels (RGB) even if image is RGBA (common with transparent PNGs)
        if not isinstance(image, np.ndarray):
            image = np.array(image.convert("RGB"))
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Resize to 224x224
        img_resized = cv2.resize(image, (224, 224))
        
        # EfficientNetB0 expectations: [0, 255] or its own preprocess_input
        # In Keras, efficientnet.preprocess_input is often a no-op as scaling is inside the model,
        # but we use it for correctness.
        img_float = img_resized.astype(np.float32)
        img_preprocessed = keras.applications.efficientnet.preprocess_input(img_float)
        
        # Add batch dimension
        return np.expand_dims(img_preprocessed, axis=0)

    def predict(self, preprocessed_img):
        """
        Performs inference on the image.
        Returns: predicted_class_index, confidence
        """
        # Note: In a real app, you would load weights here. 
        # For this demonstration, it will output random predictions if untrained.
        preds = self.model.predict(preprocessed_img)
        class_idx = np.argmax(preds[0])
        confidence = preds[0][class_idx]
        return class_idx, confidence

if __name__ == "__main__":
    # Test model building
    factory = NutritionModel()
    print("Model summary:")
    factory.model.summary()
