"""
NLP Module: Text Classification for Vehicle/Plate Type
Covers: U3-T1 Text Classification Pipeline
        U2-T2 TF-IDF, Distributed Representations
        U2-T1 Vector Space Models, Bag of Words
        U3-T3 Deep learning for Text Classification
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import pickle
import os


# Training data: (plate_number_string, vehicle_type)
# In a real system, this would be a much larger dataset
TRAINING_DATA = [
    # Cars (4-wheelers) — typically longer series
    ("KA01AB1234", "Car"), ("MH12CD5678", "Car"), ("DL08EF9012", "Car"),
    ("KA02GH3456", "Car"), ("TN09IJ7890", "Car"), ("KA50KL1234", "Car"),
    ("GJ01MN5678", "Car"), ("RJ14OP9012", "Car"), ("KA53QR3456", "Car"),
    ("UP32ST7890", "Car"), ("KL07UV1234", "Car"), ("AP28WX5678", "Car"),
    # Bikes (2-wheelers) — often single letter series
    ("KA01A1234", "Bike"), ("MH12B5678", "Bike"), ("DL08C9012", "Bike"),
    ("KA02D3456", "Bike"), ("TN09E7890", "Bike"), ("KA50F1234", "Bike"),
    ("GJ01G5678", "Bike"), ("RJ14H9012", "Bike"), ("KA53J3456", "Bike"),
    # Commercial vehicles — often with specific district codes
    ("KA01AB1234", "Commercial"), ("MH04CD5678", "Commercial"),
    ("DL01EF9012", "Commercial"), ("KA01GH3456", "Commercial"),
    # BH Series (Bharat Series for transferable vehicles)
    ("22BH1234AA", "BH_Series"), ("23BH5678BB", "BH_Series"),
    ("24BH9012CC", "BH_Series"), ("22BH3456DD", "BH_Series"),
]


def featurize_plate(plate: str) -> str:
    """
    U2-T1: Convert plate number to feature string for TF-IDF.
    Creates character n-gram features.
    """
    plate = plate.upper().replace(' ', '').replace('-', '')
    
    features = []
    
    # State code feature
    if len(plate) >= 2:
        features.append(f"STATE_{plate[:2]}")
    
    # Length feature
    features.append(f"LEN_{len(plate)}")
    
    # Series length (middle letters)
    match = re.match(r'^[A-Z]{2}[0-9]{2}([A-Z]+)[0-9]{4}$', plate)
    if match:
        series = match.group(1)
        features.append(f"SERIES_LEN_{len(series)}")
        features.append(f"SERIES_{series}")
    
    # BH series indicator
    if 'BH' in plate:
        features.append("BH_SERIES")
    
    # Character composition
    alpha_count = sum(1 for c in plate if c.isalpha())
    digit_count = sum(1 for c in plate if c.isdigit())
    features.append(f"ALPHA_{alpha_count}")
    features.append(f"DIGIT_{digit_count}")
    
    return ' '.join(features)


class PlateClassifier:
    """
    U3-T1: Text classification pipeline using TF-IDF + Naive Bayes.
    Classifies plate into vehicle types.
    """
    
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 2),  # U2-T1: Bag of N-grams
                max_features=500
            )),
            ('clf', MultinomialNB(alpha=0.1))
        ])
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self._train()
    
    def _train(self):
        """Train on the built-in training data."""
        plates, labels = zip(*TRAINING_DATA)
        features = [featurize_plate(p) for p in plates]
        
        encoded_labels = self.label_encoder.fit_transform(labels)
        self.pipeline.fit(features, encoded_labels)
        self.is_trained = True
    
    def classify(self, plate: str) -> dict:
        """
        Classify a plate number into vehicle type.
        Returns predicted class and confidence.
        """
        if not self.is_trained or not plate:
            return {"vehicle_type": "Unknown", "confidence": 0.0, "probabilities": {}}
        
        feature = featurize_plate(plate)
        proba = self.pipeline.predict_proba([feature])[0]
        predicted_idx = np.argmax(proba)
        predicted_label = self.label_encoder.inverse_transform([predicted_idx])[0]
        
        class_probs = {
            self.label_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(proba)
        }
        
        return {
            "vehicle_type": predicted_label,
            "confidence": float(proba[predicted_idx]),
            "probabilities": class_probs
        }


# Singleton classifier
_classifier = None


def get_classifier() -> PlateClassifier:
    global _classifier
    if _classifier is None:
        _classifier = PlateClassifier()
    return _classifier


def classify_plate(plate: str) -> dict:
    """Main entry point for plate classification."""
    clf = get_classifier()
    return clf.classify(plate)


def vectorize_plate_tfidf(plates: list) -> np.ndarray:
    """
    U2-T2: TF-IDF vectorization for a list of plates.
    Returns the TF-IDF matrix for visualization/analysis.
    """
    features = [featurize_plate(p) for p in plates]
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(features)
    return matrix.toarray(), vectorizer.get_feature_names_out()
