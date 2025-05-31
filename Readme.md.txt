Sentiment Analysis using TF-IDF and Dense Neural Network
This project performs binary sentiment classification on a cleaned dataset of text reviews using TF-IDF (Term Frequency–Inverse Document Frequency) vectorization combined with a Dense Neural Network. The goal is to empirically evaluate TF-IDF as an input representation compared to alternatives like Word2Vec and Bag of Words (BoW), using the same model framework for consistency.

Dataset
A labeled dataset of text reviews (e.g., Amazon product reviews).

Each sample consists of:

cleaned_text: Preprocessed textual data (lowercased, tokenized, stopword-removed, and lemmatized).

label: Binary target (0 = negative, 1 = positive sentiment).

Preprocessing
Text normalization: lowercase conversion, punctuation removal

Stopword removal using NLTK

Lemmatization with spaCy

Vectorization using TF-IDF with max_features=5000

Model Architecture
A simple feedforward neural network is used in place of LSTM due to TF-IDF producing non-sequential feature vectors:

Input Layer: TF-IDF vector of dimension 5000

Dense Layer 1: 128 neurons with ReLU activation

Dense Layer 2: 64 neurons with ReLU activation

Output Layer: 1 neuron with Sigmoid activation

Compiled with:

Loss function: binary_crossentropy

Optimizer: adam

Evaluation metric: accuracy

Training Details
Train-test split: 80% training, 20% testing

Batch size: 128

Epochs: 5

Validation split: 10% of training set

Results
Accuracy: 0.866

Training Time: 7.23 seconds

Testing Time: 0.82 seconds

These results indicate that TF-IDF is a competitive representation technique for binary sentiment classification when used with a Dense Neural Network.

File Structure
bash
Copy
Edit
├── tfidf_sentiment_analysis.ipynb   # Main notebook
├── cleaned_reviews.csv              # Input dataset (optional placeholder)
└── README.md                        # Project documentation
Conclusion
This implementation demonstrates that TF-IDF, despite being a classical method, can still achieve strong performance in sentiment classification tasks when paired with a well-tuned dense neural network. The accuracy is comparable to modern embeddings like Word2Vec, making it a strong baseline in empirical evaluations.