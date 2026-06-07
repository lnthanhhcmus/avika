import numpy as np

class Aggregator:
    def __init__(self, strategy=1):
        if strategy not in [1, 2, 3, 4]:
            raise ValueError(
                f"Invalid strategy: {strategy}. Must be 1 (Weighted RRF), 2 (Average Similarity), 3 (Majority Voting), or 4 (Max Similarity)"
            )
        
        self.strategy = strategy
        if strategy == 1:
            self.strategy_name = "Weighted RRF"
        elif strategy == 2:
            self.strategy_name = "Average Similarity"
        elif strategy == 3:
            self.strategy_name = "Majority Voting"
        else:
            self.strategy_name = "Max Similarity"
    
    def aggregate(self, sim_matrices):
        if self.strategy == 1:
            return self.weighted_rrf_aggregation(sim_matrices)
        elif self.strategy == 2:
            return self.average_aggregation(sim_matrices)
        elif self.strategy == 3:
            return self.true_majority_voting(sim_matrices)
        else:  # strategy == 4
            return self.max_aggregation(sim_matrices)

    def max_aggregation(self, sim_matrices):
        return np.max(sim_matrices, axis=0)
    
    def average_aggregation(self, sim_matrices):
        avg_sim = np.mean(sim_matrices, axis=0)
        return avg_sim
    
    def weighted_rrf_aggregation(self, sim_matrices):
        k_plus_1, n_queries, n_videos = sim_matrices.shape
        
        # Configure weights for query variants
        weights = [1.0]
        
        # Option 1: Harmonic decay weights
        for i in range(1, k_plus_1):
            weights.append(0.5 / i)

        # # Option 2: Fixed weights for enriched queries
        # if k_plus_1 > 1:
        #     # Shared Weight
        #     if k_plus_1 == 3:
        #         aug_weight = 1.0 / (k_plus_1 - 1)
        #     elif k_plus_1 == 7:
        #         aug_weight = 1.5 / (k_plus_1 - 1)
        #     elif k_plus_1 == 11:
        #         aug_weight = 2.0 / (k_plus_1 - 1)
        #     else:
        #         # Expected weight for enriched query (Example: 0.4)
            
        #     for i in range(1, k_plus_1):
        #         weights.append(aug_weight)
        
        # Smoothing constant for RRF
        k_smooth = 1.0
        
        # Initialize final score matrix
        final_score_matrix = np.zeros((n_queries, n_videos))
        
        # Process each query variant with its weight
        for idx, sim_matrix in enumerate(sim_matrices):
            w = weights[idx]
            ranks = np.argsort(np.argsort(-sim_matrix, axis=1), axis=1)
            score = w * (1.0 / (k_smooth + ranks + 1))
            final_score_matrix += score
        
        return final_score_matrix
    
    def true_majority_voting(self, sim_matrices):
        k_plus_1, n_queries, n_videos = sim_matrices.shape
        vote_counts = np.zeros((n_queries, n_videos))
        
        for idx in range(k_plus_1):
            sim_matrix = sim_matrices[idx]
            top1_videos = np.argmax(sim_matrix, axis=1)
            vote_counts[np.arange(n_queries), top1_videos] += 1
        
        sim_original = sim_matrices[0]
        tie_breaker_weight = 1e-4
        final_score_matrix = vote_counts + (tie_breaker_weight * sim_original)
        
        return final_score_matrix


if __name__ == "__main__":
    print("Aggregator strategies:")
    for s in [1, 2, 3, 4]:
        print(f"  {s}: {Aggregator(strategy=s).strategy_name}")
