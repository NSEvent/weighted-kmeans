import os
import math
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def find_euclidian_distance(xycoor1, xycoor2):
	"""Return Euclidian distance between two (x, y) coordinates."""
	d_sq = (xycoor2[0] - xycoor1[0])**2 + (xycoor2[1] - xycoor1[1])**2
	return math.sqrt(d_sq)


def find_total_distance(xlist, ylist, labels, centroids):
	"""Return total distance between xy and centroid, prints centroid count."""
	centroids_count = []
	for _ in centroids:
		centroids_count.append(0)

	# import pdb; pdb.set_trace()
	xylist = list(zip(xlist, ylist))

	total_distance = 0
	for xycoor, c_index in zip(xylist, labels):
		centroid_coor = centroids[c_index]
		centroids_count[c_index] += 1
		total_distance += find_euclidian_distance(xycoor, centroid_coor)

	print(centroids_count)
	return total_distance


MIN_CLUSTERS = 4
MAX_CLUSTERS = 9

# Read in location data and weights
cols = ['x', 'y', 'weight']
data = pd.read_csv('location_data.csv', names=cols)

xcoor0 = data.x.tolist()
ycoor0 = data.y.tolist()

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

	# Write centroid info
	with open(f'results/N={num_clusters}-centroids.csv', 'w') as fout:
		fout.write('CENTROID COORDINATES:\n')
		fout.write('centroid_num, x, y\n')
		for i, c in enumerate(centroids):
			fout.write(f'{i}, {round(c[0], 3)}, {round(c[1], 3)}\n')

	# Write client info
	with open(f'results/N={num_clusters}-results.csv', 'w') as fout:
		fout.write('CLIENT COORDINATES AND CENTROID NUMBER\n')
		fout.write('x, y, centroid_no\n')
		for x, y, n in zip(xcoor0, ycoor0, kmeans.labels_[0:len(xcoor0)-1]):
			fout.write(f'{round(x, 3)}, {round(y, 3)}, {n}\n')

	# Save clusters plot
	plt.scatter(df['x'], df['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
	# plt.savefig(f'results/N={num_clusters}-grouped.png')

	# Save clusters with centroids plot
	plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
	plt.savefig(f'results/N={num_clusters}-grouped-with-center.png')
	plt.close()

	total_distance = find_total_distance(xcoor0, ycoor0, kmeans.labels_[0:len(xcoor0)-1], centroids)
	print(total_distance)
