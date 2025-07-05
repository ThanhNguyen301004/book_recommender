import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
import os

# Định nghĩa lớp Autoencoder (phải khớp với kiến trúc lúc huấn luyện)
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

# Kiểm tra thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tải dữ liệu
file1 = "data/books_all(1-5).csv"
file2 = "data/books_all(6-10).csv"
file3 = "data/books_all(11-15).csv"
file4 = "data/books_all(16-20).csv"
file5 = "data/books_all(21-25).csv"
file6 = "data/books_all(26-30).csv"
# Đọc từng file vào DataFrame
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)
df5 = pd.read_csv(file5)
df6 = pd.read_csv(file6)
# Gộp hai DataFrame lại
df = pd.concat([df1, df2, df3, df4, df5, df6], ignore_index=True)
latent_reps = np.load('latent_reps.npy')

# Tải mô hình
input_dim = 2257  # Match the input dimension of the trained model
model = Autoencoder(input_dim=input_dim)
model.load_state_dict(torch.load('models/autoencoder_model.pth', map_location=device))
model.eval()
model.to(device)

# Hàm đề xuất sách
def recommend_books(book_title, num_recommendations=9):
    book_row = df[df['title'].str.lower() == book_title.lower()]
    if book_row.empty:
        return "Book not found", None, None
    
    idx = book_row.index[0]
    sim_scores = cosine_similarity([latent_reps[idx]], latent_reps)[0]
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]  # Top N excluding self
    
    book_indices = [i for i, _ in sim_scores]
    recommended_books = df.iloc[book_indices][['title', 'image_book', 'description', 'url', 'genres']]
    return recommended_books.to_dict('records')

# Hàm chính để chạy trên terminal
def main():
    while True:
        book_title = input("Nhập tên sách (hoặc 'exit' để thoát): ")
        if book_title.lower() == 'exit':
            print("Đã thoát chương trình.")
            break
        
        recommendations = recommend_books(book_title)
        if isinstance(recommendations, str):
            print(recommendations)
        else:
            print(f"\nCác sách được đề xuất cho '{book_title}':")
            print("recommendations", len(recommendations))

            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec['title']}")
                print(f"   Bìa sách: {rec['image_book']}...") 

if __name__ == "__main__":
    main()