
# Vision Transformer Paper Replication

A PyTorch implementation and replication of the seminal paper **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"** by Dosovitskiy et al. (2021).

## 📄 Paper Overview

This project replicates the Vision Transformer (ViT) architecture, which revolutionized computer vision by successfully applying the Transformer architecture - originally designed for natural language processing - to image classification tasks.

**Original Paper**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

## 🎯 Key Features

- **Complete ViT Implementation**: From-scratch implementation of the Vision Transformer architecture
- **Patch Embedding**: Converts images into sequences of patches for transformer processing
- **Multi-Head Self-Attention**: Core attention mechanism for capturing global dependencies
- **Position Embeddings**: Learnable position encodings for spatial awareness
- **Classification Head**: Final MLP layer for image classification
- **Training Pipeline**: End-to-end training and evaluation workflow

## 🏗️ Architecture Components

### Vision Transformer (ViT) Pipeline:
1. **Image Patching**: Divide input image into fixed-size patches (16x16)
2. **Linear Projection**: Flatten and linearly embed each patch
3. **Position Embedding**: Add learnable positional information
4. **Transformer Encoder**: Stack of Multi-Head Self-Attention and MLP layers
5. **Classification**: Use [CLS] token representation for final prediction

### Key Architecture Details:
- **Patch Size**: 16x16 pixels
- **Embedding Dimension**: Configurable (typically 768)
- **Number of Layers**: Configurable (6, 12, or 24 layers)
- **Attention Heads**: Configurable (typically 12)
- **MLP Hidden Dimension**: 4x embedding dimension

## 🛠️ Implementation

### Core Components Implemented:
- `PatchEmbedding`: Converts image patches to embeddings
- `MultiHeadSelfAttention`: Self-attention mechanism
- `MLP`: Feed-forward network with GELU activation
- `TransformerEncoderLayer`: Single transformer block
- `VisionTransformer`: Complete ViT model

### Model Variants:
- **ViT-Base**: 12 layers, 768 hidden dim, 12 heads
- **ViT-Large**: 24 layers, 1024 hidden dim, 16 heads
- **ViT-Huge**: 32 layers, 1280 hidden dim, 16 heads

## 📚 Dataset

The implementation includes training and evaluation on:
- **FoodVision_mini**

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch torchvision matplotlib numpy tqdm
```

### Usage

1. **Clone the repository**:
```bash
git clone https://github.com/NANDAGOPALNG/Vision_Transformer_Paper_Replication.git
cd Vision_Transformer_Paper_Replication
```

2. **Run the notebook**:
```bash
jupyter notebook Vision_Transformers_paper_replicating.ipynb
```

3. **Train the model**:
The notebook includes complete training loops with:
- Data preprocessing and augmentation
- Model initialization and configuration
- Training with proper learning rate scheduling
- Validation and testing procedures

## 📊 Results

### Performance Metrics:
- **Training Accuracy**: Tracked per epoch
- **Validation Accuracy**: Evaluated on held-out data
- **Loss Curves**: Both training and validation loss
- **Attention Visualizations**: Head-specific attention patterns

### Expected Results:
- CIFAR-10: ~85-90% accuracy (depending on training configuration)
- Convergence typically achieved within 50-100 epochs
- Attention maps showing meaningful spatial patterns

## 🔬 Key Insights from Replication

1. **Data Efficiency**: ViT requires substantial data or pre-training to perform well
2. **Attention Patterns**: Model learns to attend to semantically relevant regions
3. **Scalability**: Performance improves with model size and data quantity
4. **Inductive Bias**: Less built-in spatial bias compared to CNNs

## 📈 Training Details

### Hyperparameters:
- **Learning Rate**: 3e-4 (with cosine scheduling)
- **Batch Size**: 512 (or largest feasible for hardware)
- **Optimizer**: AdamW with weight decay
- **Epochs**: 100-300 (dataset dependent)
- **Warmup**: Linear warmup for first 10k steps

### Data Augmentation:
- Random resized crops
- Horizontal flips
- Color jittering
- Normalization to ImageNet statistics

## 🔍 Visualization Features

The notebook includes:
- **Attention Map Visualization**: See what the model focuses on
- **Patch Embedding Analysis**: Understanding learned representations
- **Training Metrics**: Loss and accuracy curves
- **Model Architecture Diagrams**: Visual representation of the network

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional datasets and benchmarks
- Model efficiency optimizations
- Advanced training techniques
- Comparison with CNN architectures

## 📖 References

```bibtex
@article{dosovitskiy2021image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and others},
  journal={International Conference on Learning Representations},
  year={2021}
}
```

## 📞 Contact

**Author**: NANDAGOPALNG  
**Repository**: [Vision_Transformer_Paper_Replication](https://github.com/NANDAGOPALNG/Vision_Transformer_Paper_Replication)

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

*This implementation is for educational purposes and paper replication. For production use, consider using well-established libraries like timm or transformers.*
