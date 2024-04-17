import os
import torch
import random
import numpy as np
import gradio as gr
import io
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse 

stable_diff_version = "stabilityai/stable-diffusion-2-base"
clip_version = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
gpt_version = "gpt2"
start_string = "An image of "
device = "cpu"

top_k = 2500
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

app = FastAPI()

def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained(gpt_version)
    model = GPT2LMHeadModel.from_pretrained(gpt_version, pad_token_id=tokenizer.eos_token_id)
    print("Loaded model with {} parameters".format(model.num_parameters()))
    model = model.eval()

    clip_model = CLIPModel.from_pretrained(clip_version)
    processor = CLIPProcessor.from_pretrained(clip_version)
    print("Loaded CLIP model with {} parameters".format(clip_model.num_parameters()))
    clip_model = clip_model.eval()
    return model, tokenizer, clip_model, processor

model, tokenizer, clip_model, processor = load_model()
model = model.to(device).eval()
clip_model = clip_model.to(device).eval()

def batch_clip_embeddings(possible_phrases, clip_model, processor, batch_size=32):
    clip_embs = []
    for i in range(0, len(possible_phrases), batch_size):
        batch = possible_phrases[i : i + batch_size]
        clip_in = processor(text=batch, return_tensors="pt", padding=True).to(device)
        clip_emb = clip_model.get_text_features(**clip_in)
        clip_embs.append(clip_emb)
    return torch.cat(clip_embs, dim=0)

@app.websocket("/describe")
async def describe_image(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            image_bytes = await websocket.receive_bytes()
            image_bytes_io = io.BytesIO(image_bytes)
            image = Image.open(image_bytes_io)
            clip_img_in = processor(images=image, return_tensors="pt", padding=True).to(device)
            clip_img_emb = clip_model.get_image_features(**clip_img_in)
            print("Clip image embedding shape:", clip_img_emb.shape)

            for text in generate_with_top_k_sampling(
                model,
                start_string,
                tokenizer,
                top_k,
                clip_model,
                processor,
                clip_img_emb,
                max_length=6,
                verbose=False,
                ret_last=True,
                greedy=True,
            ):
                await websocket.send_text(text)

    except WebSocketDisconnect:
        print("WebSocket connection closed")

def generate_with_top_k_sampling(
    model,
    input_text,
    tokenizer,
    top_k,
    clip,
    clip_processor,
    tgt_emb,
    max_length=100,
    verbose=False,
    ret_last=False,
    greedy=False,
):
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    output_sequence = input_ids

    with torch.no_grad():
        for _ in range(max_length):
            # Get the logits from the last layer of the model
            logits = model(output_sequence.to(device)).logits
            next_token_logits = logits[:, -1, :]

            # Get the top-k tokens for the original weights
            orig_top_k = torch.topk(next_token_logits, k=top_k, dim=-1)
            orig_top_k_tokens = orig_top_k.indices

            # Produce all k possible versions of the sequence
            possible_seqs = output_sequence.repeat(top_k, 1)
            possible_seqs = torch.cat(
                [possible_seqs, orig_top_k_tokens.reshape(-1, 1)], dim=1
            )
            possible_phrases = [tokenizer.decode(ps) for ps in possible_seqs]

            # Get the CLIP embeddings for all k possible versions of the sequence
            clip_emb = batch_clip_embeddings(
                possible_phrases, clip, clip_processor, 128
            )

            # Compute the cosine similarity between the image and all k possible versions of the sequence
            cos_sim = torch.nn.CosineSimilarity(dim=-1)
            sim = cos_sim(clip_emb, tgt_emb)
            if verbose:
                print("Clip similarities:")
                print({pf: s for pf, s in zip(possible_phrases, sim)})

            # If greedy, take the highest similarity
            if greedy:
                next_token_ind = torch.argmax(sim)
                next_token = orig_top_k_tokens[0][next_token_ind.item()]
                if verbose:
                    print("\nGreedy next token index:", next_token_ind)
                    print("Greedy next token:", tokenizer.decode(next_token))

            else:
                # Transform the similarity into a probability distribution
                # Use temperature to control the sharpness of the distribution
                sm_clip = torch.softmax(sim / 0.05, dim=-1).unsqueeze(0)

                sm = sm_clip
                if verbose:
                    print("\nTop k tokens (softmax * clip):")
                    for i in torch.argsort(sm[0]):
                        print(tokenizer.decode(orig_top_k_tokens[0][i]), sm[0][i])

                # Sample from the top-k tokens
                next_token_ind = torch.multinomial(sm, num_samples=1)
                print("Next token index:", next_token_ind)
                next_token = orig_top_k_tokens[0][next_token_ind.item()]
                if verbose:
                    print("\nNext token index:", next_token_ind)
                    print("Next token:", tokenizer.decode(next_token))

            # Concatenate the sampled token to the output sequence
            output_sequence = torch.cat(
                [output_sequence, next_token.reshape(1, 1)], dim=1
            )
            # yield tokenizer.decode(output_sequence[0])

            yield tokenizer.decode(output_sequence[0])

    generated_text = tokenizer.decode(output_sequence[0])
    if ret_last:
        return generated_text, tokenizer.decode(next_token)
    return generated_text

app.mount("/", StaticFiles(directory="app/static", html=True), name="static")