from numpy import array, exp, argsort, random

def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
	"""
	Apply softmax function to logits vector.

	Args:
		logits: (list[float]): Score vector of each token.
		temparature (float): Entropy of the output probability.

	Return:
		list[float]: A propability list of each token.
	"""
	e = exp(array(logits) / temperature)
	return e / e.sum()

def top_k(probs: list[float], k : int):
	"""
	Apply topK function to a list of probablility.

	Args:
		probs (list[float]): Selection probability list of token 
		k (int): Token poll size

	Return:
		[int, dict(int: float)]: the pick and the token poll containing ids and probality of each token
	"""

	indexs = argsort(probs)[-k:]
	filtered_probs = probs[indexs]
	filtered_probs /= sum(filtered_probs)
	pick = random.choice(indexs, p = filtered_probs)
	return [pick, {i: p for i, p in zip(indexs, filtered_probs)}] 