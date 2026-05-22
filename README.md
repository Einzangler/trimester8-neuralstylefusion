# Neural Style Fusion

## Objective
The aim of this project is to implement the **Neural Style Transfer** algorithm — one of the most exciting applications of Convolutional Neural Networks. The goal is to blend a content image and a style reference image such that the output looks like the content image painted in the style of the reference.

![Overview](https://github.com/tphanir/NeuralStyleFusion/assets/125972587/9110fc55-247a-4dca-b8c5-960ac96f7fc1)

## Approach
1. Built foundational understanding of Deep Learning and Artificial Neural Networks.
2. Implemented a Two Layer Neural Network from scratch using NumPy to classify digits on the MNIST dataset.
3. Applied optimization algorithms and hyperparameter tuning to improve the network.
4. Rebuilt the network using the PyTorch framework for better efficiency.
5. Studied CNN architectures and implemented the Neural Style Transfer algorithm using VGG-19.

## The Algorithm
The technique takes two images — a **Content image (C)** and a **Style image (S)** — and generates a new image (G) that preserves the content of C while adopting the artistic style of S.

### Cost Function

$$J(G) = \alpha \cdot J_{\text{content}}(C, G) + \beta \cdot J_{\text{style}}(S, G)$$

![Cost Function](https://github.com/tphanir/NeuralStyleFusion/assets/125972587/0d61916c-8038-40e6-9121-792a1c0344e7)

### How it works
- **Content Loss** — measures how similar the generated image is to the content image using feature maps from `conv4_2` of VGG-19

$$J_{\text{content}}(C, G) = \frac{1}{2} \sum_{i,j} (a^{[l](C)}_{ij} - a^{[l](G)}_{ij})^2$$

- **Style Loss** — measures how similar the style is using Gram Matrices computed from 5 layers of VGG-19

$$J_{\text{style}}(S, G) = \sum_{l} \lambda^{[l]} \frac{1}{(2n_H n_W n_C)^2} \| G^{[l](S)} - G^{[l](G)} \|_F^2$$

- The **Gram Matrix** captures correlations between feature maps, mathematically encoding the style of an image

$$G_{kk'} = \sum_{i,j} a^{[l]}_{ijk} \cdot a^{[l]}_{ijk'}$$

### VGG-19 Architecture

![VGG-19](https://github.com/tphanir/NeuralStyleFusion/assets/125972587/0b94c4b7-4559-46f5-bf1b-1eb90567836f)

- Feature maps from layers marked in **red** capture the style
- Feature map from the layer marked in **blue** captures the content

## Web Interface
A Flask-based web interface is included for easy use. Upload your content and style images, adjust parameters, and download the result.

🔗 **Live Demo**: [huggingface.co/spaces/prateek14/neural-style-fusion](https://huggingface.co/spaces/prateek14/neural-style-fusion)

## Installation & Usage

```bash
git clone https://github.com/Einzangler/trimester8-neuralstylefusion.git
cd trimester8-neuralstylefusion
pip install -r requirements.txt
python3 app.py
```

Then open `http://localhost:7860` in your browser.

## Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| α (Content Weight) | Controls how much content is preserved | 100 |
| β (Style Weight) | Controls how strong the style effect is | 40 |
| Iterations | Number of optimization steps | 300 |
| Image Size | Resolution of the output image | 256px |

## Results

One can infer that if the **Content** and **Style** images are chosen appropriately, the result may turn out to be a fantastic and artistically pleasing image.

### Result 1 — Building + Starry Night
<p>
  <img src="https://github.com/Einzangler/trimester8-neuralstylefusion/blob/main/IMG_0155.jpg?raw=true" width="350"/>
  <img src="https://github.com/Einzangler/trimester8-neuralstylefusion/blob/main/IMG_0147.jpg?raw=true" width="150"/>
  <img src="https://github.com/Einzangler/trimester8-neuralstylefusion/blob/main/output1.jpg?raw=true" width="350"/>
</p>

### Result 2 — Lion + Mosaic
<p>
  <img src="https://github.com/Einzangler/trimester8-neuralstylefusion/blob/main/IMG_0158.jpg?raw=true" width="350"/>
  <img src="https://github.com/Einzangler/trimester8-neuralstylefusion/blob/main/IMG_0159.jpg?raw=true" width="150"/>
  <img src="https://github.com/Einzangler/trimester8-neuralstylefusion/blob/main/output2.jpg?raw=true" width="350"/>
</p>

### Result 3 — Owl + Orange Swirl
<p>
  <img src="https://github.com/Einzangler/trimester8-neuralstylefusion/blob/main/IMG_0160.jpg?raw=true" width="350"/>
  <img src="https://github.com/Einzangler/trimester8-neuralstylefusion/blob/main/IMG_0161.jpg?raw=true" width="150"/>
  <img src="https://github.com/Einzangler/trimester8-neuralstylefusion/blob/main/output3.jpg?raw=true" width="350"/>
</p>

### Result 4 — NYC Bridge + Oil Painting
<p>
  <img src="https://github.com/Einzangler/trimester8-neuralstylefusion/blob/main/IMG_0156.jpg?raw=true" width="350"/>
  <img src="https://github.com/Einzangler/trimester8-neuralstylefusion/blob/main/IMG_0157.jpg?raw=true" width="150"/>
  <img src="https://github.com/Einzangler/trimester8-neuralstylefusion/blob/main/output4.jpg?raw=true" width="350"/>
</p>

## Tools & Technologies
- Python
- NumPy
- PyTorch
- VGG-19 (Pretrained)
- PIL
- Flask
- Matplotlib
