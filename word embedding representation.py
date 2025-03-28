import marimo

__generated_with = "0.11.28"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.decomposition import PCA
    import marimo as mo
    import gensim.downloader as api
    import umap
    import plotly.express as px
    import random
    import altair as alt
    from sklearn.manifold import TSNE

    def get_model():
        return api.load('glove-wiki-gigaword-100')

    model = get_model()
    return PCA, TSNE, alt, api, get_model, mo, model, np, pd, px, random, umap


@app.cell
def _(mo):
    mo.md(f"""
    # Word embeddings
    """)
    return


@app.cell
def _(model, np):
    topics = [
        'quantum', 'galaxy', 'symphony', 'philosophy', 'animal', 'reading'
    ]
    # Retrieve 10 closest words for each topic
    words = []
    labels = []  

    for topic in topics:
        words.append(topic)
        labels.append(topic)

        similar_words = model.most_similar(topic, topn=20)
        for word, _ in similar_words:
            words.append(word)
            labels.append(topic)

    # Get embeddings for the selected words
    embeddings = np.array([model[word] for word in words])
    return embeddings, labels, similar_words, topic, topics, word, words


@app.cell(hide_code=True)
def _2drepre(TSNE, alt, embeddings, labels, mo, pd, words):
    # Reduce dimensionality using t-SNE for better separation
    X_embedded_2d = TSNE(n_components=2, random_state=42).fit_transform(embeddings)

    # Create a DataFrame for the 2D embeddings, the words, and the topic labels
    embedding_df_2d = pd.DataFrame({
        'x': X_embedded_2d[:, 0],
        'y': X_embedded_2d[:, 1],
        'word': words,
        'topic': labels
    }).reset_index()

    # Create the chart with Marimo, adding color based on topic
    chart_2d = alt.Chart(embedding_df_2d).mark_text(
        align='center', baseline='middle', fontSize=10, fontWeight='bold'
    ).encode(
        x='x:Q',
        y='y:Q',
        text='word:N',
        color='topic:N',  # Color by topic
        tooltip=['word:N', 'topic:N']
    ).interactive().properties(width=800, height=800)  # Enable zooming and set size

    # Display the chart_2d
    plot_2d = mo.ui.altair_chart(chart_2d)

    mo.md(f"""
    <br/><br/><br/>
    ## 2D Word Embedding Visualization (TSNE)

    This script visualizes **word embeddings** using **t-SNE** for dimensionality reduction.

    1. **Select topics** `['quantum', 'galaxy', 'symphony', 'philosophy', 'animal', 'reading']`
    2. **Find similar words** (top 20 for each topic using an embedding model).
    3. **Extract embeddings** for these words.
    4. **Apply t-SNE** to project embeddings into 2D space.
    5. **Visualize**

    {plot_2d}
    """)
    return X_embedded_2d, chart_2d, embedding_df_2d, plot_2d


@app.cell(hide_code=True)
def _(embeddings, labels, mo, pd, px, umap, words):
    umap_model = umap.UMAP(n_components=3, n_neighbors=15, random_state=42)
    X_embedded_3d = umap_model.fit_transform(embeddings)

    # Create a DataFrame for the 3D embeddings, the words, and the topic labels
    embedding_df_3d = pd.DataFrame({
        'x': X_embedded_3d[:, 0],
        'y': X_embedded_3d[:, 1],
        'z': X_embedded_3d[:, 2],
        'word': words,
        'topic': labels
    }).reset_index()

    # Create a 3D scatter plot with Plotly
    fig_3d = px.scatter_3d(embedding_df_3d, x='x', y='y', z='z', color='topic',
                        text='word', 
                        # title="3D Word Embedding Visualization (UMAP)",
                        labels={'word': 'Word'},
                        opacity=0.8)

    plot_3d = mo.ui.plotly(fig_3d)

    # Show the plot_3d
    mo.md(f"""
    <br/><br/>
    ## 3D Word Embedding Visualization (UMAP)

    This script visualizes **word embeddings** using **t-SNE** for dimensionality reduction. Key steps:

    1. **Select topics** `['quantum', 'galaxy', 'symphony', 'philosophy', 'animal', 'reading']`
    2. **Find similar words** (top 20 for each topic using an embedding model).
    3. **Extract embeddings** for these words.
    4. **Apply UMAP** to project embeddings into 3D space.
    5. **Visualize**

    {plot_3d}
    """)
    return X_embedded_3d, embedding_df_3d, fig_3d, plot_3d, umap_model


@app.cell
def _(mo, model):
    def print_similarity(word1, word2):
        similarity = model.similarity(word1, word2)
        return f"Similarity between `{word1}and and {word2}`: **{similarity:.4f}**"

    mo.md(f"""
    <br/><br/>
    ## Distance between two words

    This function calculates **cosine similarity** between two words using a pre-trained word embedding model:

    - **Higher values** (closer to 1) mean the words are semantically similar.
    - **Lower values** (closer to 0) indicate no relation.
    - Examples:
        - {print_similarity("cat", "dog")}
        - {print_similarity("cat", "car")}
    """)
    return (print_similarity,)


@app.cell
def _(mo, model):
    def show_analogy(positive, negative, expected=None):
        result = model.most_similar(positive=positive, negative=negative, topn=5)

        positive_str = " + ".join(f"`{word}`" for word in positive)
        negative_str = " - ".join(f"`{word}`" for word in negative)

        if expected and expected in model:
            top_words = [word for word, _ in result]
            if expected in top_words:
                rank = top_words.index(expected) + 1
                return f"| {positive_str} - {negative_str} | **`{result[0][0]}∗∣** | {expected}` | {rank} | {', '.join([f'{word}:{score:.2f}' for word, score in result])} |"
            else:
                similarity = model.similarity(result[0][0], expected)
                return f"| {positive_str} - {negative_str} | **`{result[0][0]}∗*∣ | {expected}` | - | Sim: {similarity:.4f} | {', '.join([f'{word}:{score:.2f}' for word, score in result])} |"

        return f"| {positive_str} - {negative_str} | **`{result[0][0]}`** | - | - | {', '.join([f'{word}:{score:.2f}' for word, score in result])} |"

    mo.md(f"""
    <br/><br/>
    ## Word Analogy Examples 

    | Analogy | Top Result | Expected | Rank | Details |
    |---------|------------|----------|------|---------|
    {show_analogy(positive=["king", "woman"], negative=["man"], expected="queen")}
    {show_analogy(positive=["paris", "italy"], negative=["france"], expected="rome")}
    {show_analogy(positive=["dollar", "europe"], negative=["america"], expected="euro")}
    """)
    return (show_analogy,)


if __name__ == "__main__":
    app.run()
