Terrorism-Related Text Classification Using LSTM
Overview
A web-based text classification application that analyses user-provided text or the content of a webpage and classifies it into one of three categories: Safe, Suspicious, or Dangerous.

The application combines an LSTM-based sequence-processing branch with Bag-of-Words (BoW) features to support text classification. A Flask backend exposes endpoints for webpage and direct-text analysis.
Features
- Text classification: Analyse submitted text and return a predicted category.
- URL analysis: Fetch and extract readable content from a webpage before classification.
- Text preprocessing: Lowercase text, remove non-alphabetic characters, normalize whitespace, and filter common stop words.
- Hybrid feature representation: Use tokenized sequences for the LSTM branch and CountVectorizer features for the BoW branch.
- Prediction output: Return the predicted category, confidence score, and class probabilities.
- Health endpoint: Check application and model-loading status.
Model Architecture
The training pipeline builds a two-input neural network:
1. Sequence branch: Tokenized text is padded to a fixed length, passed through an embedding layer, and processed by an LSTM layer.
2. BoW branch: CountVectorizer features are passed through a dense layer and dropout.
3. Classification head: The two representations are concatenated and passed through dense and dropout layers, followed by a three-class softmax output.
The model is trained with the Adam optimizer and categorical cross-entropy loss. Early stopping and learning-rate reduction callbacks are used during training.
Technology Stack
- Python
- TensorFlow / Keras
- Flask
- Pandas and NumPy
- Scikit-learn
- NLTK
- Requests
- Beautiful Soup
Project Files
File	Description
app.py	Flask application for webpage and text analysis, model loading, preprocessing, and prediction.
train.py	Prepares text features, builds and trains the LSTM + BoW model, and saves the trained artifacts.
dataset_creation.py	Generates a labelled dataset with Safe, Suspicious, and Dangerous text categories.
fetcher.py	Retrieves webpage content, extracts meaningful text, and cleans it before analysis.
terrorism_dataset.csv	Dataset used by the training pipeline.
terrorism_detection_model.h5	Saved trained Keras model.
tokenizer.pickle	Saved text tokenizer.
vectorizer.pickle	Saved Bag-of-Words vectorizer.
label_encoder.pickle	Saved label encoder.
requirements.txt	Python dependencies and pinned package versions.
