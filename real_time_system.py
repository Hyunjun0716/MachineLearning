import joblib
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
import string
import time


# Check device compatibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Best BERT Parameters
learning_rate = 3.3874006317859486e-05
batch_size = 37
epochs = 1

bert_tokenizer = BertTokenizer.from_pretrained('/kaggle/input/bert-model/bert_tokenizer')
tokenizer = joblib.load('/kaggle/input/lstm-bilstm-cnn-model/tokenizer.pkl')
best_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
best_bert.load_state_dict(torch.load('/kaggle/input/bert-model/bert_model.pth', map_location=device, weights_only=True), strict=False)
best_bert.eval()

# Define PyTorch LSTM Model
class LSTMClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=5000, embedding_dim=embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        x = self.fc(hn[-1])
        return self.sigmoid(x).view(-1)

# Define PyTorch BiLSTM Model
class BiLSTMClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_layers, dropout):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=5000, embedding_dim=embed_dim)
        self.bilstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, 
            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.bilstm(x)
        x = self.fc(torch.cat((hn[-2], hn[-1]), dim=1))
        return self.sigmoid(x).view(-1)    


# Define PyTorch CNN Model
class CNNClassifier(nn.Module):
    def __init__(self, embed_dim, num_filters, kernel_size, dropout):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=5000, embedding_dim=embed_dim)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.mean(x, dim=2)
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x).view(-1)    
# Initialize model with the same hyperparameters
best_lstm = LSTMClassifier(
    embed_dim=137,     
    hidden_dim=225,
    num_layers=2,
    dropout=0.28019970078781725
).to(device)

state_dict = torch.load('/kaggle/input/lstm-bilstm-cnn-model/lstm_model.pth', map_location=device, weights_only=True)
best_lstm.load_state_dict(state_dict)
best_lstm.eval()

# Load BiLSTM Model
best_bilstm = BiLSTMClassifier(
    embed_dim=137,
    hidden_dim=225,
    num_layers=2,
    dropout=0.28019970078781725
).to(device)

state_dict_bilstm = torch.load('/kaggle/input/lstm-bilstm-cnn-model/bilstm_model.pth', map_location=device, weights_only=True)
best_bilstm.load_state_dict(state_dict_bilstm)
best_bilstm.eval()

# Load CNN Model
best_cnn = CNNClassifier(
    embed_dim=137,
    num_filters=225,
    kernel_size=4,
    dropout=0.28019970078781725
).to(device)

state_dict_cnn = torch.load('/kaggle/input/lstm-bilstm-cnn-model/cnn_model.pth', map_location=device, weights_only=True)
best_cnn.load_state_dict(state_dict_cnn)
best_cnn.eval()

vectorizer = joblib.load('/kaggle/input/traditional-ml/vectorizer.pkl')
LR = joblib.load('/kaggle/input/traditional-ml/logistic_regression.pkl')
RFC = joblib.load('/kaggle/input/traditional-ml/random_forest.pkl')
SVM = joblib.load('/kaggle/input/traditional-ml/svm.pkl')
NB = joblib.load('/kaggle/input/traditional-ml/naive_bayes.pkl')

# Preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# SHAP masker and wrapper
masker = shap.maskers.Text(bert_tokenizer)

class BERTWrapper:
    def __call__(self, texts):
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        elif isinstance(texts, str):
            texts = [texts]
        tokens = bert_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            logits = best_bert(**tokens).logits
            probs = torch.nn.functional.softmax(logits, dim=1)
        return probs[:, 1].detach().cpu().numpy()

# Full Prediction Function
def predict_news(news):
    total_start_time = time.time()
    news = clean_text(news)

    results = {}
    timings = {}

    # BERT
    start_time = time.time()
    inputs = bert_tokenizer(news, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        output = best_bert(**inputs).logits
        pred_bert = torch.argmax(output, dim=1).item()
    end_time = time.time()
    results["BERT Prediction"] = "🔴 FAKE NEWS" if pred_bert == 0 else "🟢 REAL News"
    timings["BERT"] = end_time - start_time

    # Tokenization for DL models
    seq = tokenizer.texts_to_sequences([news])
    padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    tensor_input = torch.tensor(padded, dtype=torch.long, device=device)

    # LSTM
    start_time = time.time()
    with torch.no_grad():
        pred_lstm = best_lstm(tensor_input).item()
    end_time = time.time()
    results["LSTM Prediction"] = "🔴 FAKE NEWS" if pred_lstm < 0.5 else "🟢 REAL News"
    timings["LSTM"] = end_time - start_time

    # BiLSTM
    start_time = time.time()
    with torch.no_grad():
        pred_bilstm = best_bilstm(tensor_input).item()
    end_time = time.time()
    results["BiLSTM Prediction"] = "🔴 FAKE NEWS" if pred_bilstm < 0.5 else "🟢 REAL News"
    timings["BiLSTM"] = end_time - start_time

    # CNN
    start_time = time.time()
    with torch.no_grad():
        pred_cnn = best_cnn(tensor_input).item()
    end_time = time.time()
    results["CNN Prediction"] = "🔴 FAKE NEWS" if pred_cnn < 0.5 else "🟢 REAL News"
    timings["CNN"] = end_time - start_time

    # Traditional ML models
    vec_input = vectorizer.transform([news])

    start_time = time.time()
    pred_lr = LR.predict(vec_input)[0]
    end_time = time.time()
    results["Logistic Regression Prediction"] = "🔴 FAKE NEWS" if pred_lr == 0 else "🟢 REAL News"
    timings["Logistic Regression"] = end_time - start_time

    start_time = time.time()
    pred_rfc = RFC.predict(vec_input)[0]
    end_time = time.time()
    results["Random Forest Prediction"] = "🔴 FAKE NEWS" if pred_rfc == 0 else "🟢 REAL News"
    timings["Random Forest"] = end_time - start_time

    start_time = time.time()
    pred_nb = NB.predict(vec_input)[0]
    end_time = time.time()
    results["Naive Bayes Prediction"] = "🔴 FAKE NEWS" if pred_nb == 0 else "🟢 REAL News"
    timings["Naive Bayes"] = end_time - start_time

    start_time = time.time()
    pred_svm = SVM.predict(vec_input)[0]
    end_time = time.time()
    results["SVM Prediction"] = "🔴 FAKE NEWS" if pred_svm == 0 else "🟢 REAL News"
    timings["SVM"] = end_time - start_time

    # Voting
    fake_count = sum(1 for result in results.values() if "FAKE NEWS" in result)

    if fake_count >= 6:
        final_result = "🔴 This news is likely FAKE NEWS."
    elif 3 <= fake_count < 6:
        final_result = "🟡 Careful, we are not sure about this news."
    else:
        final_result = "🟢 This news is likely REAL News (Not Fake News)."

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    results["Final Prediction"] = final_result
    results["Prediction Time (s)"] = timings
    results["Total Time"] = total_time

    # SHAP if final prediction is fake or warning
    if "FAKE" in final_result or "Careful" in final_result:
        print("\n[Explanation] SHAP explanation based on BERT...")
        start_time = time.time()
        explainer = shap.Explainer(BERTWrapper(), masker=masker)
        shap_values = explainer([news])
        shap.plots.text(shap_values[0])
        shap_time = time.time() - start_time
        results["SHAP Explanation Time"] = shap_time
    else:
        shap_time = 0.0

    total_time = time.time() - total_start_time + shap_time
    results["Prediction Time (s)"] = timings
    results["Total Time"] = total_time

    return results

# Example run
if __name__ == "__main__":
    news = input("\nEnter news text: ")
    predictions = predict_news(news)

    print("\n=== Prediction Result ===")
    for key in predictions:
        if key not in ["Prediction Time (s)", "Final Prediction", "Total Time", "SHAP Explanation Time"]:
            print(f"{key:35}: {predictions[key]}")

    print("\n=== Model Timing (in seconds) ===")
    for model, timing in predictions["Prediction Time (s)"].items():
        print(f"{model:25}: {timing:.4f}s")
    if "SHAP Explanation Time" in predictions:
        print(f"SHAP Explanation Time         : {predictions['SHAP Explanation Time']:.4f}s")

    print("\n=== Final Decision ===")
    print(f"{predictions['Final Prediction']}")
    print(f"Total Time                        : {predictions['Total Time']:.4f}s")
