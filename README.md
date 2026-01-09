# VPI Prediction Using GAT

This project aims to predict Visceral Pleural Invasion (VPI) using a **Graph Attention Network (GAT)**. The model pipeline consists of three main components:

1. **Transformer**  
2. **MLP (Multi-Layer Perceptron)**  
3. **Fusion of Transformer and MLP features**  
4. **GAT for final prediction**

## Model Architecture

The feature extraction and prediction workflow is as follows:

ROI (Region of Interest, 128^3)  
↓  
Node Modeling (8^3 per node)
→ shape: (4096, 512)  
↓  
Graph Construction  
↓  
GAT Prediction


- **Node Modeling:** Each ROI of size 128³ is divided into nodes of size 8³, resulting in 4096 nodes, each represented by a 512-dimensional feature vector.  
- **Graph Construction:** The graph is built based on the node features and then passed into a GAT for prediction.  

## Training Issue

The model **failed to converge** due to incorrect graph modeling. The core problem is:

- By constructing a graph in this way, the model **implicitly assumes that the structure of all ROIs is identical**.  
- Even slight differences in ROI size or structure violate this assumption, which fundamentally undermines the graph representation.  
- As a result, the GAT is not learning meaningful relationships, leading to training failure.

## Notes

- Careful attention must be paid to how graphs are constructed from 3D ROIs to avoid assumptions of uniformity.  
- Alternative approaches may include adaptive graph construction or size-normalized node features to handle ROI variability.
