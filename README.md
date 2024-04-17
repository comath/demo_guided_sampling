# Demo: image description

AI Village demo for guided sampling of a generative LM using multimodal embeddings to describe images.

This demo shows how to guide the sampling of an autoregressive language model (GPT-2) in order to minimize the distance to the embedding of a target image. The optimization is carried out in the embedding space of a multi-modal encoder model (CLIP). The resulting text is a description of the image.

## Setup

Install pytorch for your system. Then install the requirements with:
```bash 
pip install -r requirements.txt
``` 

## Usage

Generate images with the notebook `generate_images.ipynb`.

Then run the notebook `demo_image_description.ipynb` to generate image descriptions.
