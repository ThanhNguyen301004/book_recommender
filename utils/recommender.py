from sklearn.metrics.pairwise import cosine_similarity

def recommend_games(game_name, df, latent_reps, num_recommendations=9):
    # Tìm dòng tương ứng với tên game
    game_row = df[df['title'].str.lower() == game_name.lower()]
    if game_row.empty:
        return []

    idx = game_row.index[0]

    # Tính điểm tương đồng cosine
    sim_scores = cosine_similarity([latent_reps[idx]], latent_reps)[0]
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Lấy các chỉ số gợi ý (bỏ qua chính game gốc)
    sim_scores = sim_scores[1:num_recommendations + 1]
    
    game_indices = [i[0] for i in sim_scores if i[0] < len(df)]
    print(f"Recommended game indices: {game_indices}")
    # Trả về các cột phù hợp
    return df.iloc[game_indices][['title', 'image_book', 'description', 'genres', 'url']].to_dict('records')
