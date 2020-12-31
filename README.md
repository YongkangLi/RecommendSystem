# RecommendSystem

## Implemented
  - Collaborative Filtering
  - Content Based
  - Pearson Similarity
  - MinHash
  - Jaccard Similarity

## Todo
  - Speed up.
  - Content-Based MiHash shoud be theoretically faster than Content-Based, which is not the actual case. Trying to figure it out.

## Usage:
```
  $ cd src
  $ ./collaborative-filtering-recommend.py	# Collaborative Filtering User-User Recommendation via Pearson Similarities
  $ ./collaborative-filtering-minhash.py	# Collaborative Filtering User-User Recommendation via Jaccard Similarities computed through MinHash
  $ ./collaborative-filtering-relative.py	# Collaborative Filtering User-User Recommendation via Pearson Similarity but relative scale
  $ ./content-based-recommend.py		# Content-Based Recommendation via Cosine Similarites
  $ ./content-based-minhash.py			# Content-Based Recommendation via Jaccard Similarites computed through Minhash
```

