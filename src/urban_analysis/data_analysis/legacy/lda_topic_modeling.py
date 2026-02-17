import os
import json
import MeCab
import unidic_lite
import pandas as pd
import gensim
from gensim import corpora
from gensim.models import CoherenceModel
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import japanize_matplotlib

# --- Configuration ---
CONFIG = {
    "input_json_path": "./data/processed/poi/filtered_facilities.json",
    "output_csv_path_template": "data/processed/lda/lda_results_cl_{num_topics}.csv",
    "lda_params": {
        "num_topics": 10,
        "random_state": 100,
        "update_every": 1,
        "chunksize": 100,
        "passes": 10,
        "alpha": "auto",
        "per_word_topics": True,
    },
    "dictionary_filter": {
        "no_below": 10,
        "no_above": 0.5,
    },
    "visualization": {
        "wordcloud_words": 20,
        "bargraph_words": 10,
    },
    "pos_to_keep": ['名詞', '副詞','形容詞', '形容動詞'], # Parts of speech to keep
}


def load_and_prepare_data(file_path: str) -> tuple[list[str], list[str]]:
    """
    Loads facility data from a JSON file and extracts descriptions and names.

    Args:
        file_path: Path to the input JSON file.

    Returns:
        A tuple containing two lists: descriptions and facility names.
    """
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    descriptions = []
    facility_names = []
    for item in data:
        desc_text = ""
        if item.get('description'):
            desc_text = " ".join(item['description']) if isinstance(item['description'], list) else item['description']
        elif item.get('description_short'):
            desc_text = item['description_short']
        descriptions.append(desc_text)
        facility_names.append(item.get('name', 'Unnamed Facility'))

    print(f"Processed {len(descriptions)} facilities.")
    return descriptions, facility_names


def tokenize_documents(descriptions: list[str], pos_to_keep: list[str]) -> list[list[str]]:
    """
    Tokenizes a list of documents using MeCab, keeping specific parts of speech.

    Args:
        descriptions: A list of strings (documents) to tokenize.
        pos_to_keep: A list of parts of speech to retain.

    Returns:
        A list of lists, where each inner list contains the tokens for a document.
    """
    print("Tokenizing descriptions...")
    mecab = MeCab.Tagger(f"-d {unidic_lite.DICDIR}")

    def tokenize(text: str) -> list[str]:
        """Tokenizes a single text string."""
        tokens = []
        node = mecab.parseToNode(text)
        while node:
            features = node.feature.split(',')
            part_of_speech = features[0]
            if part_of_speech in pos_to_keep:
                base_form = features[6] if len(features) > 6 and features[6] != '*' else node.surface
                tokens.append(base_form)
            node = node.next
        return tokens

    tokenized_docs = [tokenize(desc) for desc in tqdm(descriptions)]
    print("Tokenization complete.")
    if tokenized_docs:
        print(f"--- Example Tokens for the first facility:\n{tokenized_docs[0]}")
    return tokenized_docs


def train_lda_model(tokenized_docs: list[list[str]], lda_params: dict, filter_params: dict) -> tuple[gensim.models.LdaModel, list, corpora.Dictionary]:
    """
    Trains an LDA model on the tokenized documents.

    Args:
        tokenized_docs: A list of tokenized documents.
        lda_params: Dictionary of parameters for the LdaModel.
        filter_params: Dictionary of parameters for filtering the dictionary.

    Returns:
        A tuple containing the trained LDA model, the corpus (BoW), and the dictionary.
    """
    print("\nCreating dictionary and corpus...")
    dictionary = corpora.Dictionary(tokenized_docs)
    
    print("Filtering dictionary extremes...")
    dictionary.filter_extremes(**filter_params)
    
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    print(f"Dictionary size: {len(dictionary)}")
    print(f"Number of documents in corpus: {len(corpus)}")

    print("Training LDA model...")
    lda_model = gensim.models.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        **lda_params
    )
    print("LDA model training complete.")
    
    print("\n--- Top 10 words for each topic ---")
    for topic in lda_model.print_topics(num_words=10):
        print(topic)
        
    return lda_model, corpus, dictionary


def evaluate_model(lda_model: gensim.models.LdaModel, tokenized_docs: list[list[str]], dictionary: corpora.Dictionary):
    """
    Calculates and prints the coherence score of the LDA model.

    Args:
        lda_model: The trained LDA model.
        tokenized_docs: The list of tokenized documents.
        dictionary: The Gensim dictionary.
    """
    print("\nCalculating Coherence Score...")
    coherence_model_lda = CoherenceModel(
        model=lda_model, 
        texts=tokenized_docs, 
        dictionary=dictionary, 
        coherence='c_v'
    )
    coherence_lda = coherence_model_lda.get_coherence()
    print(f"Coherence Score (c_v): {coherence_lda}")


def save_topic_results(lda_model: gensim.models.LdaModel, corpus: list, facility_names: list[str], descriptions: list[str], output_path_template: str):
    """
    Saves the dominant topic for each document to a CSV file.

    Args:
        lda_model: The trained LDA model.
        corpus: The document-term matrix (BoW).
        facility_names: List of facility names.
        descriptions: List of original descriptions.
        output_path_template: The template for the output CSV file path.
    """
    print("\nSaving results to CSV...")
    results = []
    for i, doc_bow in enumerate(corpus):
        topic_distribution = lda_model.get_document_topics(doc_bow, minimum_probability=0.0)
        dominant_topic = sorted(topic_distribution, key=lambda x: x[1], reverse=True)[0][0]
        topic_words = lda_model.show_topic(dominant_topic, topn=10)
        topic_keywords = ", ".join([word for word, prop in topic_words])
        
        results.append({
            'facility_name': facility_names[i],
            'dominant_topic': dominant_topic,
            'topic_keywords': topic_keywords,
            'description': descriptions[i]
        })

    df_results = pd.DataFrame(results)
    
    num_topics = lda_model.num_topics
    output_csv_path = output_path_template.format(num_topics=num_topics)
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    df_results.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"Results saved to {output_csv_path}")
    print("\n--- First 5 rows of the results ---")
    print(df_results.head())


def visualize_topics(lda_model: gensim.models.LdaModel, vis_params: dict):
    """
    Generates and displays word clouds and bar graphs for each topic.

    Args:
        lda_model: The trained LDA model.
        vis_params: Dictionary of parameters for visualization.
    """
    print("\n--- Visualizing Topics ---")
    
    font_path = ""
    try:
        font_path = japanize_matplotlib.get_font_path()
        if not os.path.exists(font_path):
            print(f"Warning: Font file not found at {font_path}. Word Clouds may not display Japanese characters correctly.")
    except Exception as e:
        print(f"Could not get japanize_matplotlib font path: {e}")

    # Word Clouds
    print("\nGenerating Word Clouds...")
    for i, topic in lda_model.show_topics(formatted=False, num_words=vis_params['wordcloud_words']):
        topic_dict = {word: prob for word, prob in topic}
        try:
            wc = WordCloud(font_path=font_path,
                           background_color="white",
                           width=800, height=400,
                           max_words=vis_params['wordcloud_words'],
                           prefer_horizontal=0.9).generate_from_frequencies(topic_dict)
            plt.figure(figsize=(10, 5))
            plt.imshow(wc, interpolation='bilinear')
            plt.axis("off")
            plt.title(f"Topic {i} Word Cloud")
            plt.show()
        except Exception as e:
            print(f"Error generating Word Cloud for Topic {i}: {e}")
            if font_path: print(f"Attempted font path: {font_path}")

    # Bar Graphs
    print("\nGenerating Bar Graphs...")
    for i, topic in lda_model.show_topics(formatted=False, num_words=vis_params['bargraph_words']):
        fig, ax = plt.subplots(figsize=(10, 5))
        words = [word for word, prob in topic]
        probs = [prob for word, prob in topic]
        
        ax.barh(words, probs, color='skyblue')
        ax.set_title(f"Topic {i} Top Words")
        ax.invert_yaxis()
        plt.xlabel("Probability")
        plt.ylabel("Word")
        plt.tight_layout()
        plt.show()

    print("\nVisualization complete.")


def main():
    """
    Main function to run the LDA topic modeling pipeline.
    """
    # 1. Load and prepare data
    descriptions, facility_names = load_and_prepare_data(CONFIG["input_json_path"])

    # 2. Tokenize documents
    tokenized_docs = tokenize_documents(descriptions, CONFIG["pos_to_keep"])

    # 3. Train LDA model
    lda_model, corpus, dictionary = train_lda_model(
        tokenized_docs, 
        CONFIG["lda_params"], 
        CONFIG["dictionary_filter"]
    )

    # 4. Evaluate model
    evaluate_model(lda_model, tokenized_docs, dictionary)

    # 5. Save results
    save_topic_results(
        lda_model, 
        corpus, 
        facility_names, 
        descriptions, 
        CONFIG["output_csv_path_template"]
    )

    # 6. Visualize topics
    visualize_topics(lda_model, CONFIG["visualization"])


if __name__ == "__main__":
    main()
