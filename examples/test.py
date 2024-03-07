from predictors import *

predictor = NNModelPredictor(model_path=r"weights\DNN_acc0.91_07-03-2024_21-26.h5",
                             vectorizer_path=r"weights\DNN_acc0.91_07-03-2024_21-26_tokenizer.json",
                             debug_mode=False)

NB_predictor = SimpleModelPredictor(model_path=r"weights\NB_f1_0.791_07-03-2024_21-26.pkl",
                             vectorizer_path=r"weights\NB_f1_0.791_07-03-2024_21-26_vectorizer.pkl",
                             debug_mode=False)

LR_predictor = SimpleModelPredictor(model_path=r"weights\LR_f1_0.876_07-03-2024_21-26.pkl",
                             vectorizer_path=r"weights\LR_f1_0.876_07-03-2024_21-26_vectorizer.pkl",
                             debug_mode=False)

# v1_predictor = NNModelPredictor(model_path=r"weights\DNN_f1_acc0.94.h5",
#                              vectorizer_path=r"weights\DNN_f1_acc0.94_tokenizer.json",
#                              debug_mode=False)

# v2_predictor = NNModelPredictor(model_path=r"weights\DNN_acc0.92.h5",
#                              vectorizer_path=r"weights\DNN_acc0.92_tokenizer.json",
#                              debug_mode=False)

test_phrases = ['I love you man, I really like your style', 'I hate life', "Wow this is amazing!",
                'i feel my portfolio demonstrates how eager i am to learn but some who know me better might call it annoyingly persistent']


predictors = [predictor, NB_predictor, LR_predictor]

for predictor in predictors:
    for phrase in test_phrases:
        print(predictor.get_prediction(phrase), sep=" ")
    print()