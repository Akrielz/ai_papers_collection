# AI Papers Collection

## Description

! Warning: This is a work in progress. !

This is a list of papers that I've read or plan to read. 

I've also included links to implementations of the papers that I've found 
online, or my own implementations if I've done them, along with personal
notes and tags

## Tags

The available tags are: 
- `#CV` for computer vision
- `#NLP` for natural language processing
- `#BIO` for bioinformatics
- `#AUDIO` for audio processing
- `#AL` for active learning
- `#RL` for reinforcement learning
- `#GM` for generative models
- `#LLM` for large language models

## List

### You Only Look Once: Unified, Real-Time Object Detection
- Alternative Title: YOLOv1
- Link: https://arxiv.org/abs/1506.02640
- Own Implementation: Work in progress
- Alternative Implementation:
  - https://github.com/tanjeffreyz/yolo-v1
  - https://github.com/lovish1234/YOLOv1
- Personal Notes:
  - Core idea: Split the image into a grid of cells, and for each cell predict
    the bounding boxes and class probabilities for the objects in that cell.
  - Flaws: The class probabilities aren't per bounding box, but per cell. This
    means that if there are multiple objects of different classes, we will 
    predict only one class for the cell, and the other objects will be ignored.
- Tags: `#CV`


### Segment Anything
- Alternative Title: SAM
- Link: https://arxiv.org/abs/2304.02643
- Core Idea:  
  The model is split into three parts: 
  - a small and fast prompt encoder which encodes the prompt, regardless of 
  its form (points, rectangles, shapes, text).
  - a large and slow image encoder which encodes the image. This pre-computes
  the features for the image, and is only computed once.
  - a lightweight decoder which combines the prompt and image encodings to
  predict the segmentation mask.  
  
  The model is trained in 3 stages:
  - pre-training on a large dataset available online
  - constantly retraining on more data, adnotated by the human experts, helped
    by the model's predictions
  - fully autonomous training, where the model is trained on the data 
    annotated by the model itself
- Tags: `#CV`
  

### A Generalist Agent
- Alternative Title: GATO
- Link: https://arxiv.org/abs/2205.06175
- Core Idea: Use a Transformer architecture, with the purpose of the agent
  multiple tasks. Due to the input being very different from dataset to dataset
  (e.g. images, text, audio), the model is split into the following inputs
  categories: Proprioception (for robots), Image, Text, Continuous Action,
  Discrete Action. All these inputs are converted to tokens which are used
  on very specific positions. The model is trained on multiple tasks, and
  the tasks are weighted based on their importance.
- Tags: `#CV` `#NLP` `#AUDIO` `#RL`



### An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- Alternative Title: Vision Transformer (ViT)
- Link: https://arxiv.org/abs/2010.11929
- Own Implementation: https://github.com/Akrielz/vision_models_playground/blob/main/vision_models_playground/models/classifiers/vision_transformer.py
- Alternative Implementation:
  - https://github.com/lucidrains/vit-pytorch
- Core idea:  
  Use a transformer architecture for image classification. 
  The image is split into patches, and the patches are flattened into a 
  sequence of tokens. The transformer is then applied to the sequence of 
  tokens.
- Advantages:  
  The more data you have, the better the model performs. 
  The model is also very flexible, and can be applied to many different 
  tasks.
- Flaws:  
  For small datasets, the model performs worse than CNNs. The model 
  also requires a lot of memory, and is slow to train.
- Tags: `#CV`


### Attention Is All You Need
- Alternative Title: Transformer
- Link: https://arxiv.org/abs/1706.03762
- Own Implementation: https://github.com/Akrielz/vision_models_playground/blob/main/vision_models_playground/components/attention/transformer.py
- Official Implementation: https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer
- Alternative Implementation:
  - https://github.com/lucidrains/x-transformers
- Core idea: Introduces Attention Layer. The input of Attention is split into
  queries, keys and values, and the attention is calculated between the 
  queries and keys. The output is then calculated as a weighted sum of the
  values, where the weights are the attention scores. This allows the model
  to learn which tokens from the input are related to what tokens from the
  output.
- Flaws: The attention layer is very expensive to compute.
- Tags: `#NLP` `#LLM` `#GM`


### Generative Adversarial Networks
- Alternative Title: GAN
- Link: https://arxiv.org/abs/1406.2661
- Own Implementation: https://github.com/Akrielz/vision_models_playground/blob/main/vision_models_playground/models/generative/adverserial/gan.py
- Core idea: Train a generator and a discriminator. The generator tries to
  generate images that look real, and the discriminator tries to distinguish
  between real and fake images. The generator is trained to fool the 
  discriminator, and the discriminator is trained to not be fooled by the
  generator.
- Flaws: Hard to train. Since the generator is updated based on the 
  discriminator's output, if the discriminator is too good, the generator
  will never be able to fool it, and will never learn. If the discriminator
  is too bad, the generator will learn to fool it, but the generator will
  not learn to generate realistic images.
- Tags: `#GM` `#CV`


### Deep Residual Learning for Image Recognition:
- Alternative Title: ResNet
- Link: https://arxiv.org/abs/1512.03385
- Own Implementation: https://github.com/Akrielz/vision_models_playground/blob/main/vision_models_playground/models/classifiers/resnet.py
- Official Implementations: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
- Core idea: Use residual blocks to train very deep neural networks. The 
  residual blocks allow the network to learn the identity function, which
  helps with training, as you no longer have to worry about the optimal depth
  of the network.
- Tags: `#CV`

