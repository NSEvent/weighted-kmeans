import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

MIN_CLUSTERS = 4
MAX_CLUSTERS = 9

# Read in location data and weights
cols = ['x', 'y', 'weight']
data = pd.read_csv('location_data.csv', names=cols)

xcoor = data.x.tolist()
ycoor = data.y.tolist()
weights = data.weight.tolist()

# Adjust x and y coors for weights
# Weights are interpreted like frequency
for i, wt in enumerate(weights):
	for _ in range(1, wt):
		xcoor.append(xcoor[i])
		ycoor.append(ycoor[i])

data = {'x': xcoor, 'y': ycoor}
df = pd.DataFrame(data, columns=['x', 'y'])

if not os.path.exists('results'):
	os.mkdir('results')
# Iterate through different num_clusters
for num_clusters in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
	kmeans = KMeans(n_clusters=num_clusters).fit(df)
	centroids = kmeans.cluster_centers_
	with open(f'results/N={num_clusters}-centroids.txt', 'w') as fout:
		fout.write('CENTROID COORDINATES (x, y):\n')
		fout.write(str(centroids))
	
	# Save clusters plot
	plt.scatter(df['x'], df['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
	plt.savefig(f'results/N={num_clusters}-grouped.png')
	
	# Save clusters with centroids plot
	plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
	plt.savefig(f'results/N={num_clusters}-grouped-with-center.png')
	plt.close()
