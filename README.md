# Image-Search-Engine
This project will build a simple image search engine based-on Spectral Clustering Algorithm.
Let talk about the idea of this project:
We need to extract the feature (color, shape,..) from image and convert it to vector. Then, we make similarity graph from these vectors with each vector as a node. 
After that, we use Spectral Clustering to split all node in to different cluster. When we want to search image, we extract the image to find their feature and compute the
distance from the image to the center of all cluster. We'll select the nearest cluster, and return some of the nearest image of this cluster.
