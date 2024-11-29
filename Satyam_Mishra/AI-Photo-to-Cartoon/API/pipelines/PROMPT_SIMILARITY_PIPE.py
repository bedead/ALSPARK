from sentence_transformers import SentenceTransformer


def load_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


def run_similarity(prompt1: str, prompt2: str, model):
    embeddings1 = model.encode(prompt1)
    embeddings2 = model.encode(prompt2)

    similarities = model.similarity(embeddings1, embeddings2)

    print(f"Score : {similarities[0][0]}")

    return similarities[0][0]


def get_better_prompt(blip_prompt: str, user_prompt: str, model, threshold: float):
    score = run_similarity(
        prompt1=blip_prompt,
        prompt2=user_prompt,
        model=model,
    )

    if score >= threshold:
        return user_prompt
    else:
        return blip_prompt


# personal testing

# model = load_model()
# blip = "a photo of man"
# user = "a photo of a man with headphone"
# print(f"Better prompt : {get_better_prompt(blip, user, model, threshold=0.7)}")
