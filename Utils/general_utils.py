import numpy as np


def convert_to_user_item(data, n_user, n_movies):
    out = []
    for u in range(n_user):
        user_index = data[:, 0] == u + 1
        ind_movie = data[user_index, 1]
        ind_rating = data[user_index, 2]
        rating = np.zeros(n_movies)

        rating[ind_movie - 1] = ind_rating

        out.append(list(rating))
    return np.array(out)
