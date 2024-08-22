# Transformers from scratch

The implemented Transformer architecture is written from scratch using PyTorch, described in ["Attention is all you need"](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).

The Vision Transformer (ViT) model, described in ["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/pdf/2010.11929), is extended from the base Transformer architecture implemented here.

For the ViT variant, I placed layer normalization inside residual connections (i.e. before attention) instead of placing them between residual blocks, which is the method used in the original Transformers paper. This technique is described in ["On Layer Normalization in the Transformer Architecture"](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf), and would lead to a smaller gradient as a result.
