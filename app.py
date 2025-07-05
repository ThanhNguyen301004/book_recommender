import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template_string, render_template

app = Flask(__name__)

# Define Autoencoder class (must match the architecture used during training)
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, input_dim), nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
file1 = "data/books_all(1-5).csv"
file2 = "data/books_all(6-10).csv"
file3 = "data/books_all(11-15).csv"
file4 = "data/books_all(16-20).csv"
file5 = "data/books_all(21-25).csv"
file6 = "data/books_all(26-30).csv"

# Read CSV files into DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)
df5 = pd.read_csv(file5)
df6 = pd.read_csv(file6)

# Concatenate DataFrames
df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
latent_reps = np.load('latent_reps.npy')

# Load model
input_dim = 2257  # Match the input dimension of the trained model
model = Autoencoder(input_dim=input_dim)
model.load_state_dict(torch.load('models/autoencoder_model.pth', map_location=device))
model.eval()
model.to(device)

# Function to recommend books
def recommend_books(book_title, num_recommendations=9):
    book_row = df[df['title'].str.lower() == book_title.lower()]
    if book_row.empty:
        return None
    
    idx = book_row.index[0]
    sim_scores = cosine_similarity([latent_reps[idx]], latent_reps)[0]
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]  # Top N excluding self
    
    book_indices = [i for i, _ in sim_scores]
    recommended_books = df.iloc[book_indices][['title', 'image_book', 'url', 'genres']]
    return recommended_books.to_dict('records')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    book_name = data['book_name']
    recommendations = recommend_books(book_name)
    return jsonify(recommendations)

@app.route('/book_names', methods=['GET'])
def book_names():
    try:
        book_list = df['title'].tolist()
        if not book_list:
            return jsonify({"error": "No book names found in data"}), 404
        return jsonify(book_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_book_info', methods=['POST'])
def get_book_info():
    data = request.get_json()
    book_name = data['book_name']
    book_row = df[df['title'].str.lower() == book_name.lower()]
    if not book_row.empty:
        return jsonify({
            'title': book_row['title'].values[0],
            'image_book': book_row['image_book'].values[0],
            'url': book_row['url'].values[0]
        })
    return jsonify({}), 404

@app.route('/random_book', methods=['GET'])
def random_book():
    random_book = df.sample(1)[['title', 'image_book', 'url']].iloc[0].to_dict()
    return jsonify(random_book)


if __name__ == '__main__':
    app.run(debug=True)