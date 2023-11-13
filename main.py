# Load Libraries
from scipy.sparse import csc_matrix, csr_matrix
import numpy as np
from sklearn.random_projection import SparseRandomProjection
from scipy.spatial.distance import cosine
from collections import defaultdict
from tqdm import tqdm
import hashlib
import argparse
import time


# Jaccard Similarity Algo

def lsh_jaccard_similarity(ratings_matrix, user_movie_sets, num_permutations=150, num_bands=5, rows_per_band=30, max_bucket_size=50, threshold=0.5):
    
    ### Minhashing
    
    # Convert to CSC format for efficient column operations
    ratings_matrix_csc = csc_matrix(ratings_matrix)

    # Generate random permutations
    num_rows = ratings_matrix_csc.shape[0]
    permutations = [np.random.permutation(num_rows) for _ in range(num_permutations)]

    # Create Minhash signatures for each user
    minhash_signatures = {}
    for user_id in tqdm(range(ratings_matrix_csc.shape[1]), desc="Computing Minhash Signatures"):
        user_column = ratings_matrix_csc[:, user_id]
        signature = []
        for perm in permutations:
            permuted_indices = perm[user_column.nonzero()[0]]
            min_index = np.min(permuted_indices) if permuted_indices.size > 0 else np.inf
            signature.append(min_index)
        minhash_signatures[user_id] = signature

    ### LSH

    # Hashing bands of signatures into buckets
    buckets = defaultdict(list)
    for user_id, signature in tqdm(minhash_signatures.items(), desc="Hashing bands to buckets"):
        for band_idx in range(num_bands):
            start_idx = band_idx * rows_per_band
            end_idx = start_idx + rows_per_band if band_idx < num_bands - 1 else len(signature)
            band = tuple(signature[start_idx:end_idx])
            bucket_key = hashlib.sha1(str(band).encode()).hexdigest()
            buckets[bucket_key].append(user_id)

    # Generating candidate pairs
    candidate_pairs = set()
    for bucket_users in tqdm(buckets.values(), desc="Generating candidate pairs"):
        if len(bucket_users) > 1 and len(bucket_users) <= max_bucket_size:
            for i in range(len(bucket_users)):
                for j in range(i + 1, len(bucket_users)):
                    if bucket_users[i] != bucket_users[j]:
                        candidate_pairs.add((bucket_users[i], bucket_users[j]))

    ### Calculate similarities

    # Compute Jaccard Similarity for Candidate Pairs
    jaccard_similarities = {}
    for user1, user2 in candidate_pairs:
        set1, set2 = user_movie_sets[user1], user_movie_sets[user2]
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        similarity = intersection / union if union != 0 else 0
        jaccard_similarities[(user1, user2)] = similarity

    # Filter Pairs Based on Jaccard Similarity Threshold
    filtered_pairs = {pair: sim for pair, sim in jaccard_similarities.items() if sim > threshold}

    return filtered_pairs, candidate_pairs, jaccard_similarities

def lsh_cosine_similarity(ratings_matrix, num_components=100, num_bands=33, rows_per_band=3, max_bucket_size=50, threshold=0.73):
    # Convert to CSR format for efficient row operations
    ratings_matrix_csr = csr_matrix(ratings_matrix)

    ### Random Projections for LSH

    # Apply random projection to reduce dimensions while preserving cosine similarity
    rp = SparseRandomProjection(n_components=num_components, random_state=42)
    reduced_data = rp.fit_transform(ratings_matrix_csr)

    ### LSH

    # Hashing the reduced data into buckets
    buckets = defaultdict(list)
    for user_id in tqdm(range(reduced_data.shape[0]), desc="Hashing users to buckets"):
        user_vector = reduced_data[user_id, :]
        for band_idx in range(num_bands):
            start_idx = band_idx * rows_per_band
            end_idx = start_idx + rows_per_band if band_idx < num_bands - 1 else num_components
            band = tuple(user_vector[start_idx:end_idx])
            bucket_key = hashlib.sha1(str(band).encode()).hexdigest()
            buckets[bucket_key].append(user_id)

    # Generating candidate pairs
    candidate_pairs = set()
    for bucket_users in tqdm(buckets.values(), desc="Generating candidate pairs"):
        if len(bucket_users) > 1 and len(bucket_users) <= max_bucket_size:
            for i in range(len(bucket_users)):
                for j in range(i + 1, len(bucket_users)):
                    candidate_pairs.add((bucket_users[i], bucket_users[j]))

    ### Calculate Cosine Similarities

    # Compute Cosine Similarity for Candidate Pairs
    cosine_similarities = {}
    for user1, user2 in candidate_pairs:
        # Convert to 1D arrays
        vector1 = reduced_data[user1, :].toarray().ravel()
        vector2 = reduced_data[user2, :].toarray().ravel()

        # Note: 1 - cosine distance is cosine similarity
        similarity = 1 - cosine(vector1, vector2)
        cosine_similarities[(user1, user2)] = similarity

    # Filter Pairs Based on Cosine Similarity Threshold
    filtered_pairs = {pair: sim for pair, sim in cosine_similarities.items() if sim > threshold}

    return filtered_pairs, candidate_pairs, cosine_similarities

def lsh_d_cosine_similarity(ratings_matrix, num_components=100, num_bands=33, rows_per_band=3, max_bucket_size=50, threshold=0.73):
    # Convert all non-zero elements of the ratings matrix to 1 for DCS
    ratings_matrix_dcs = ratings_matrix.copy()
    ratings_matrix_dcs.data = np.ones_like(ratings_matrix_dcs.data)

    ### Random Projections for LSH

    # Apply random projection to reduce dimensions
    rp = SparseRandomProjection(n_components=num_components, random_state=42)
    reduced_data = rp.fit_transform(ratings_matrix_dcs)

    ### LSH

    # Hashing the reduced data into buckets
    buckets = defaultdict(list)
    for user_id in tqdm(range(reduced_data.shape[0]), desc="Hashing users to buckets"):
        user_vector = reduced_data[user_id, :]
        for band_idx in range(num_bands):
            start_idx = band_idx * rows_per_band
            end_idx = start_idx + rows_per_band if band_idx < num_bands - 1 else num_components
            band = tuple(user_vector[start_idx:end_idx])
            bucket_key = hashlib.sha1(str(band).encode()).hexdigest()
            buckets[bucket_key].append(user_id)

    # Generating candidate pairs
    candidate_pairs = set()
    for bucket_users in tqdm(buckets.values(), desc="Generating candidate pairs"):
        if len(bucket_users) > 1 and len(bucket_users) <= max_bucket_size:
            for i in range(len(bucket_users)):
                for j in range(i + 1, len(bucket_users)):
                    candidate_pairs.add((bucket_users[i], bucket_users[j]))

    ### Calculate Discrete Cosine Similarities

    # Compute Cosine Similarity for Candidate Pairs
    discrete_cosine_similarities = {}
    for user1, user2 in candidate_pairs:
        # Convert to 1D arrays
        vector1 = reduced_data[user1, :].toarray().ravel()
        vector2 = reduced_data[user2, :].toarray().ravel()

        # Note: 1 - cosine distance is cosine similarity
        similarity = 1 - cosine(vector1, vector2)
        discrete_cosine_similarities[(user1, user2)] = similarity

    # Filter Pairs Based on Discrete Cosine Similarity Threshold
    filtered_pairs = {pair: sim for pair, sim in discrete_cosine_similarities.items() if sim > threshold}

    return filtered_pairs, candidate_pairs, discrete_cosine_similarities


def main():
    start_time = time.time()  # Record the start time

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LSH for Similarity Measures")
    parser.add_argument('-d', '--data', required=True, help="Data file path")
    parser.add_argument('-s', '--seed', type=int, required=True, help="Random seed")
    parser.add_argument('-m', '--measure', required=True, choices=['js', 'cs', 'dcs'], help="Similarity measure (js / cs / dcs)")
    args = parser.parse_args()

    # Load the data
    data = np.load(args.data)

    # Set the random seed
    np.random.seed(args.seed)

    # Check if the dataset uses 1-based indexing and convert to 0-based if necessary
    if np.min(data[:, 0]) == 1:
        data[:, 0] -= 1  # Convert user_id to 0-based index
    if np.min(data[:, 1]) == 1:
        data[:, 1] -= 1  # Convert movie_id to 0-based index

    # Create the sparse matrix
    num_users = np.max(data[:, 0]) + 1
    num_movies = np.max(data[:, 1]) + 1
    ratings_matrix = csr_matrix((data[:, 2], (data[:, 0], data[:, 1])), shape=(num_users, num_movies))

    # Create Sets of Rated Movies for Each User
    user_movie_sets = defaultdict(set)
    for user_id, movie_id, _ in data: 
        user_movie_sets[user_id].add(movie_id)

    # Process based on the chosen similarity measure
    if args.measure == 'js':
        # Jaccard Similarity
        filtered_pairs, candidate_pairs, jaccard_similarities = lsh_jaccard_similarity(
            ratings_matrix, user_movie_sets, num_permutations=100, num_bands=33, rows_per_band=3, max_bucket_size=2000, threshold=0.5)
        output_file = 'js.txt'
    elif args.measure == 'cs':
        # Cosine Similarity
        filtered_pairs, candidate_pairs, cosine_similarities = lsh_cosine_similarity(
            ratings_matrix, num_components=20, num_bands=7, rows_per_band=3, max_bucket_size=500, threshold=0.73)
        # Call the cosine similarity function here
        output_file = 'cs.txt'
    elif args.measure == 'dcs':
        # Discrete Cosine Similarity
        filtered_pairs, candidate_pairs, discrete_cosine_similarities = lsh_d_cosine_similarity(
            ratings_matrix, num_components=80, num_bands=27, rows_per_band=3, max_bucket_size=500, threshold=0.73)
        # Call the discrete cosine similarity function here
        output_file = 'dcs.txt'

    # Write the results to the output file
    with open(output_file, 'w') as f:
        for pair in filtered_pairs:
            f.write(f'{pair[0]}, {pair[1]}\n')

    end_time = time.time()  # Record the end time
    total_runtime = end_time - start_time

    # Print the total runtime
    print(f"Total runtime: {total_runtime:.2f} seconds")

if __name__ == "__main__":
    main()