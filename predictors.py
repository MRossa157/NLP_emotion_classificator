from abc import ABC

import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.models import load_model
from keras.preprocessing.text import tokenizer_from_json

class BasePredictor(ABC):
    
    def __init__(self, model_path: str, vectorizer_path: str, debug_mode = False) -> None:
        self.debug_mode = debug_mode      
        self.class_mapping = {0: 'Sadness', 1: 'Joy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}       
        
    def get_prediction(self, text: str) -> str:
        raise NotImplementedError
    
    
class SimpleModelPredictor(BasePredictor):
    def __init__(self, model_path: str, vectorizer_path: str, debug_mode=False) -> None:
        super().__init__(model_path, vectorizer_path, debug_mode)
        
        if not model_path.endswith(".pkl"):
            raise ValueError("Путь к модели должен содержать файл с расширением .pkl")
        
        if not vectorizer_path.endswith(".pkl"):
            raise ValueError("Путь к модели должен содержать файл с расширением .pkl")

        self.model: MultinomialNB  | LogisticRegression | SVC = joblib.load(model_path)
        self.vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)
    
    def get_prediction(self, text: str) -> str:
        text_vectorized = self.vectorizer.transform([text])
        predictions = self.model.predict(text_vectorized)
        if self.debug_mode:
            print(predictions)
        predicted_class_names = self.class_mapping[int(predictions)]
        return predicted_class_names
    
    
class NNModelPredictor(BasePredictor):
    def __init__(self, model_path: str, vectorizer_path: str, debug_mode=False) -> None:
        super().__init__(model_path, vectorizer_path, debug_mode)
        if not model_path.endswith(".h5"):
            raise ValueError("Путь к модели должен содержать файл с расширением .h5")
        if not vectorizer_path.endswith(".json"):
            raise ValueError("Путь к модели должен содержать файл с расширением .json")
        
        self.model: Sequential = load_model(model_path)
        self.vectorizer: Tokenizer = self.__load_tokenizer(vectorizer_path)
    
    def __load_tokenizer(self, path: str) -> Tokenizer:
        with open(path) as f:
            data = f.read()
            loaded_tokenizer = tokenizer_from_json(data)
        return loaded_tokenizer
    
    def get_prediction(self, text: str) -> str | None:
        new_sequences = self.vectorizer.texts_to_sequences(text)
        new_padded = pad_sequences(new_sequences, maxlen=100, padding='post')

        predictions = self.model.predict(new_padded)
        
        predicted_labels = np.array(predictions.argmax(axis=1))
        if self.debug_mode:
            print(predicted_labels)

        predicted_class_names = [self.class_mapping[label] for label in predicted_labels]
        return predicted_class_names[0] if predicted_class_names else None
    
    
    
if __name__ == "__main__":
    # predictor = NNModelPredictor(model_path=r"weights\DNN_f1_acc0.94.h5",
    #                              vectorizer_path=r"weights\DNN_f1_acc0.94_tokenizer.json",
    #                              debug_mode=True)
    
    NB_predictor = SimpleModelPredictor(model_path=r"weights/NB_f1_0.816.pkl",
                                 vectorizer_path=r"weights/NB_f1_0.816_vectorizer.pkl",
                                 debug_mode=False)
    
    LR_predictor = SimpleModelPredictor(model_path=r"weights\LR_f1_0.904.pkl",
                                 vectorizer_path=r"weights\LR_f1_0.904_vectorizer.pkl",
                                 debug_mode=False)
    
    DNN_predictor = NNModelPredictor(model_path=r"weights\DNN_f1_acc0.94.h5",
                                 vectorizer_path=r"weights\DNN_f1_acc0.94_tokenizer.json",
                                 debug_mode=False)
    
    test_phrases = ['I love you man, I really like your style', 'I hate life', "I'm not good"]
    
    
    predictors = [NB_predictor, LR_predictor, DNN_predictor]
    
    for predictor in predictors:
        for phrase in test_phrases:
            print(predictor.get_prediction(phrase), sep=" ")
        print()