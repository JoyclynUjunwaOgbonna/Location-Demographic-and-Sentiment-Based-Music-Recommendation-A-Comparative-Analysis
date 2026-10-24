# Location, Demographic and Sentiment-Based Music Recommendation: A Comparative Analysis

This repository contains the Jupyter notebooks for my dissertation project. The project builds a music recommendation system that predicts which songs a listener will prefer, using three kinds of information:

- **Location:** the place or activity the listener is in (gym, bedroom, library and others)
- **Demographics:** age group and gender
- **Sentiment:** the mood of the song's lyrics, classified with transformer (BERT-family) models

The project compares three classifiers, Supervised LDA, Random Forest and a Multilayer Perceptron, across three versions of the dataset. Each version uses lyric moods from a different transformer model.

## Locations

Songs were collected for eight locations:

Gym · Bedroom · Library · Office · Restaurant · Bar · Mall · Train/Bus

## Pipeline

The notebooks are listed in the order they were run.

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | [Location_Data_Collection](Location_Data_Collection.ipynb) | Finds location- or activity-labelled Spotify playlists and pulls their tracks with the Spotify API, then builds the participant questionnaire. |
| 2 | [Lyrics and Musical Feature Data Collection](Lyrics%20and%20Musical%20Feature%20Data%20Collection%20(1).ipynb) | Gets the lyrics from the Genius API and the audio features (danceability, energy, valence and others) from Spotify. |
| 3 | [Genre Data Collection](Genre%20Data%20Collection.ipynb) | Gets each song's genre through its artist ID. This step was done separately because genre is attached to the artist, not the track. |
| 4 | [Lyrics Analysis and Cleaning](Lyrics%20Analysis%20and%20Cleaning.ipynb) | Corrects lyrics that the Genius API matched to the wrong song. |
| 5 | [Lyrics_Mood_Modelling](Lyrics_Mood_Modellling%20(1).ipynb) | Classifies the mood of each song's lyrics with three pretrained transformer models (listed below). |
| 6 | [Data Processing For Modelling](Data%20Processing%20For%20Modelling.ipynb) | Processes participants' song rankings by demographic group, merges them with the song metadata, audio features, lyrics and moods, engineers features, and creates the train and test splits. |
| 7 | [Exploratory Data Analysis](Exploratory%20Data%20Analysis.ipynb) | Summary statistics, correlations, and feature distributions by song preference for each location. |
| 8 | [SupervisedLDA](SupervisedLDA.ipynb) | Builds and evaluates the Supervised LDA model. |
| 9 | [RandomForest](RandomForest.ipynb) | Builds the Random Forest classifier: a baseline, then hyperparameter tuning (number of trees, maximum depth, PCA) to reduce overfitting. |
| 10 | [MLP](MLP.ipynb) | Builds a multilayer perceptron in PyTorch with skorch: a baseline with hidden layers of 52 and 25 neurons, then tuning. |

## Mood classification models

Each model produces one version of the dataset, and every classifier is trained and evaluated on all three versions:

| Dataset | Hugging Face model |
|---------|--------------------|
| RoBERTa | [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions) |
| DistilRoBERTa | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) |
| DistilRoBERTa Mitch | [`michellejieli/emotion_text_classifier`](https://huggingface.co/michellejieli/emotion_text_classifier) |

## Evaluation

The models are evaluated separately for each demographic target group (male and female participants aged 21–25, 26–30 and 31–35) with these metrics:

- Precision@k, Average Precision@k and NDCG@k
- Precision and ROC curves
- Confusion matrices

## Tech stack

Python · pandas · NumPy · scikit-learn · PyTorch · skorch · Hugging Face Transformers · Spotipy · LyricsGenius · Matplotlib · Seaborn

## Notes

- **API credentials:** the data collection notebooks contain the placeholders `YOUR_SPOTIFY_CLIENT_ID` / `YOUR_SPOTIFY_CLIENT_SECRET`. Replace them with your own keys from the [Spotify Developer Dashboard](https://developer.spotify.com/dashboard). The Genius notebooks also need a Genius API access token.
- **Cell outputs:** `RandomForest.ipynb` is uploaded with its outputs cleared to stay under GitHub's file size limit. Run the notebook to regenerate them.
- **Large notebooks:** `MLP.ipynb` and `SupervisedLDA.ipynb` are large, so GitHub may not preview them. Download them or open them in Jupyter to view them.
- **Data:** the datasets and participant questionnaire responses are not included in this repository.
