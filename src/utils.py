from numpy import array, exp, argsort, random

def softmax(logits: list[float], temperature: float = 1):
	"""
	Apply softmax function to logits vector
	"""
	e = exp(array(logits) / temperature)
	return e / e.sum()

def top_k(probs: list[float], k : int):
	indexs = argsort(probs)[-k:]
	filtered_probs = probs[indexs]
	filtered_probs /= sum(filtered_probs)
	pick = random.choice(indexs, p = filtered_probs)
	return [pick, {i: p for i, p in zip(indexs, filtered_probs)}] 