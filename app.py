from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
from fetcher import WebContentFetcher
import os
import traceback

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production

class TerrorismClassifier:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.label_encoder = None
        self.max_length = 60  # Must match training max_length
        self.is_loaded = False

    def load_model(self):
        try:
            if os.path.exists('terrorism_detection_model.h5'):
                self.model = tf.keras.models.load_model('terrorism_detection_model.h5')
            else:
                print("Model file not found. Please train the model first.")
                return False
            
            if os.path.exists('tokenizer.pickle'):
                with open('tokenizer.pickle', 'rb') as f:
                    self.tokenizer = pickle.load(f)
            else:
                print("Tokenizer file not found.")
                return False
            
            if os.path.exists('vectorizer.pickle'):
                with open('vectorizer.pickle', 'rb') as f:
                    self.vectorizer = pickle.load(f)
            else:
                print("Vectorizer file not found.")
                return False
            
            if os.path.exists('label_encoder.pickle'):
                with open('label_encoder.pickle', 'rb') as f:
                    self.label_encoder = pickle.load(f)
            else:
                print("Label encoder file not found.")
                return False
            
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def preprocess_text(self, text):
        import re
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize

            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
            return ' '.join(tokens)
        except:
            words = text.split()
            common_stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'a', 'an'}
            filtered_words = [word for word in words if word not in common_stopwords and len(word) > 2]
            return ' '.join(filtered_words)

    def predict(self, text):
        if not self.is_loaded:
            return {"error": "Model not loaded. Please check if model files exist."}

        try:
            processed_text = self.preprocess_text(text)
            if not processed_text.strip():
                return {"error": "No meaningful text content found for analysis."}

            sequence = self.tokenizer.texts_to_sequences([processed_text])[0]
            chunks = [sequence[i:i + self.max_length] for i in range(0, len(sequence), self.max_length)]
            if len(chunks[-1]) < self.max_length:
                chunks[-1] += [0] * (self.max_length - len(chunks[-1]))
            padded_chunks = np.array(chunks)

            bow_features = self.vectorizer.transform([processed_text]).toarray()

            predictions = []
            for chunk_seq in padded_chunks:
                chunk_seq = np.expand_dims(chunk_seq, axis=0)
                pred = self.model.predict([chunk_seq, bow_features], verbose=0)
                predictions.append(pred[0])

            avg_pred = np.mean(predictions, axis=0)
            predicted_class = np.argmax(avg_pred)
            confidence = avg_pred[predicted_class]
            class_names = ['Safe', 'Suspicious', 'Dangerous']
            predicted_label = class_names[predicted_class]

            return {
                'prediction': predicted_label,
                'confidence': float(confidence),
                'probabilities': {
                    'Safe': float(avg_pred[0]),
                    'Suspicious': float(avg_pred[1]),
                    'Dangerous': float(avg_pred[2])
                },
                'processed_text_length': len(processed_text.split())
            }
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

fetcher = WebContentFetcher()
classifier = TerrorismClassifier()

if not classifier.load_model():
    print("Warning: Model not loaded. Please train the model first.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_url():
    try:
        data = request.get_json() if request.is_json else request.form
        url = data.get('url', '').strip()
        if not url:
            return jsonify({"error": "URL is required"}), 400
        fetch_result = fetcher.fetch_url_content(url)
        if "error" in fetch_result:
            return jsonify(fetch_result), 400
        processed_text = fetch_result.get('processed_text', '')
        if not processed_text or len(processed_text.strip()) < 10:
            return jsonify({
                "error": "No sufficient text content found on the webpage",
                "url": url,
                "title": fetch_result.get('title', 'Unknown')
            }), 400
        if not classifier.is_loaded:
            return jsonify({
                "error": "Classification model not loaded. Please train the model first.",
                "url": url,
                "title": fetch_result.get('title', 'Unknown'),
                "text_preview": processed_text[:200] + "..." if len(processed_text) > 200 else processed_text
            }), 500
        classification_result = classifier.predict(processed_text)
        if "error" in classification_result:
            return jsonify({
                "error": classification_result["error"],
                "url": url,
                "title": fetch_result.get('title', 'Unknown')
            }), 500
        result = {
            "url": url,
            "title": fetch_result.get('title', 'Unknown'),
            "word_count": fetch_result.get('word_count', 0),
            "prediction": classification_result['prediction'],
            "confidence": classification_result['confidence'],
            "probabilities": classification_result['probabilities'],
            "text_preview": processed_text[:300] + "..." if len(processed_text) > 300 else processed_text,
            "status": "success"
        }
        return jsonify(result)
    except Exception as e:
        print(f"Error in analyze_url: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    try:
        data = request.get_json() if request.is_json else request.form
        text = data.get('text', '').strip()
        if not text:
            return jsonify({"error": "Text is required"}), 400
        if len(text) < 10:
            return jsonify({"error": "Text too short for meaningful analysis"}), 400
        if not classifier.is_loaded:
            return jsonify({
                "error": "Classification model not loaded. Please train the model first.",
                "text_preview": text[:200] + "..." if len(text) > 200 else text
            }), 500
        classification_result = classifier.predict(text)
        if "error" in classification_result:
            return jsonify({
                "error": classification_result["error"],
                "text_preview": text[:200] + "..." if len(text) > 200 else text
            }), 500
        result = {
            "text_preview": text[:300] + "..." if len(text) > 300 else text,
            "word_count": len(text.split()),
            "prediction": classification_result['prediction'],
            "confidence": classification_result['confidence'],
            "probabilities": classification_result['probabilities'],
            "processed_text_length": classification_result.get('processed_text_length', 0),
            "status": "success"
        }
        return jsonify(result)
    except Exception as e:
        print(f"Error in analyze_text: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/health')
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": classifier.is_loaded,
        "fetcher_available": True
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
