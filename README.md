# weighted-kmeans

Clusters weighted (x, y) coordinates using k-means clustering

The data represents the location of a client and the priority (weight) that the client has. The goal of k-means clustering is calculate the coordinates to place servers in the center of k clusters of clients, so that the average squared Euclidean distance between a client and server are minimized.

Tested with Python 3.7.5 in Linux environment

## Install required dependencies
```python
pip install -r requirements.txt
```
## Run
```python
python kmeans.py
```
