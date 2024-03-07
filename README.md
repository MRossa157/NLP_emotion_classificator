# What I did
- I Loaded Textual data and preprocess and clean the texts.
- I did Tokenization and inter encoding using Tokeinzer Class.
- I did Padding .
- I Trained an each model to Get good F1-score.
- I Visualized Results.
- Did Evaluations on Test Data.

# About the Dataset:
Each entry in this dataset consists of a text segment representing a Twitter message and a corresponding label indicating the predominant emotion conveyed. 

The emotions are classified into six categories: sadness (0), joy (1), love (2), anger (3), fear (4), and surprise (5). 

Whether you're interested in sentiment analysis, emotion classification, or text mining, this dataset provides a rich foundation for exploring the nuanced emotional landscape within the realm of social media.

[Download link](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data)

## Key Features:
text: A string feature representing the content of the Twitter message.

label: A classification label indicating the primary emotion, with values ranging from 0 to 5. 

### Potential Use Cases: 
Sentiment Analysis: Uncover the prevailing sentiments in English Twitter messages across various emotions. 

Emotion Classification: Develop models to accurately classify tweets into the six specified emotion categories. 

Textual Analysis: Explore linguistic patterns and expressions associated with different emotional states. 
### Sample Data: 
text | label | |------------------------------------------------|-------| 

that was what i felt when i was finally accept…| 1 | 

i take every day as it comes i'm just focussin…| 4 | 

i give you plenty of attention even when i fee…| 0 |

# How to use
- Download Dataset and place it in the dataset folder (should result in `./dataset/text.csv`)
- Run `Model.ipynb` and execute all the code cells to train each model
- Use the `predictors.py` module to create an instance of `Predictor` and load the model weights and its vectorizer into it
- Call the `get_prediction(text: str)` method to get prediction
