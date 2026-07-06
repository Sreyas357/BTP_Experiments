class RewardTerm:
    pass


class LinearReward(RewardTerm):
    """
    a^T mu + b
    """
    def __init__(self, indices, weights, offset=0.0):
        if len(indices) != len(weights):
            raise ValueError("indices and weights must match")

        self.indices = indices
        self.weights = weights
        self.offset = offset


class MaxLinearReward(RewardTerm):
    """
    weight * max_i (a_i^T mu + b_i)
    """
    def __init__(self, vectors, offsets, weight):
        if len(vectors) != len(offsets):
            raise ValueError("vectors and offsets must match")

        self.vectors = vectors
        self.offsets = offsets
        self.weight = weight