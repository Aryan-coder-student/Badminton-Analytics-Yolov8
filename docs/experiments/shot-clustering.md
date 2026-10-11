# Experiment — Shot Embedding and Clustering

## Setup

The project implements DINOv2, CLIP, and ResNet image embedding strategies plus KMeans, DBSCAN, and HDBSCAN clustering strategies.

A scratchpad experiment processes the 1,431 extracted shot frames, caches DINOv2 embeddings, averages the three sampled frames per shot, and compares KMeans candidates.

## Recorded result

The recorded run selected six visual clusters with silhouette score `0.207529`.

## Interpretation

This experiment is a visual grouping/pre-labeling tool. It is not a semantic classifier by itself.
