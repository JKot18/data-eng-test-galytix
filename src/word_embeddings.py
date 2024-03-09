import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine, euclidean
from tqdm import tqdm
from typing import Tuple


class WordEmbeddings:
    """
    Class for working with word embeddings.

    Attributes:
        wv (gensim.models.KeyedVectors): Pretrained Word2Vec model.
    """

    def __init__(self, bin_file_path, phrases_file):
        """
        Initialize WordEmbeddings object.

        Args:
            bin_file_path (str): Path to the binary file containing the Word2Vec model.
        """
        try:
            # Load pretrained Word2Vec from binary
            self.wv = KeyedVectors.load_word2vec_format(bin_file_path, binary=True, limit=1000000)
        except FileNotFoundError:
            print(f"Error: File '{bin_file_path}' not found.")
            raise
        except Exception as e:
            print(f"Error loading vectors Word2Vec from file '{bin_file_path}': {e}")
            raise
        try:
            # Load phrases from file
            self.phrases_df = pd.read_csv(phrases_file, encoding='latin1')
        except FileNotFoundError:
            print(f"Error '{phrases_file}' not found.")
            return
        except Exception as e:
            print(f"Error on reading file '{phrases_file}': {e}")
            return

    def get_similarity(self, phrase1: str, phrase2: str) -> Tuple[float, float]:
        """
        Compute similarity between two phrases.

        Args:
            phrase1 (str): First phrase.
            phrase2 (str): Second phrase.

        Returns:
            tuple: A tuple containing cosine similarity and Euclidean similarity.
        """
        try:
            # split phrases to words and get vector reference
            words1 = phrase1.split()
            words2 = phrase2.split()

            vec1 = sum([self.wv[word] for word in words1 if word in self.wv])
            vec2 = sum([self.wv[word] for word in words2 if word in self.wv])

            # Normalizing vectors
            vec1_norm = vec1 / max(1, np.linalg.norm(vec1))
            vec2_norm = vec2 / max(1, np.linalg.norm(vec2))

            # Calculate
            cosine_similarity = 1 - cosine(vec1_norm, vec2_norm)
            euclidean_similarity = 1 / (1 + euclidean(vec1_norm, vec2_norm))

            return cosine_similarity, euclidean_similarity

        except Exception as e:
            print(f"Error calculating phrase '{phrase1}' and '{phrase2}': {e}")
            raise

    def calculate_similarity(self, phrases_file: str, output_file: str, embeddings_file: str):
        """
            Calculate similarity between phrases and save the results to a file.

            Args:
                phrases_file (str): Path to the file containing phrases.
                output_file (str): Path to the output file where results will be saved.
                embeddings_file (str): Path to the binary file containing the Word2Vec model.
            """

        try:
            # Create class exemplar to work with embeddings_file
            word_embeddings = WordEmbeddings(embeddings_file)
        except FileNotFoundError:
            print(f"Error: File '{embeddings_file}' not found.")
            return
        except Exception as e:
            print(f"Error on loading Word2Vec из file '{embeddings_file}': {e}")
            return

        # Use tqdm to track processing with progress bar
        total_iterations = len(self.phrases_df) * (len(self.phrases_df) - 1) // 2
        with tqdm(total=total_iterations) as pbar:
            similarities = []
            for i in range(len(self.phrases_df)):
                for j in range(i + 1, len(self.phrases_df)):
                    try:
                        phrase1 = self.phrases_df['Phrases'][i]
                        phrase2 = self.phrases_df['Phrases'][j]
                        cosine_sim, euclidean_sim = word_embeddings.get_similarity(phrase1, phrase2)
                        similarities.append({'Phrase 1': phrase1, 'Phrase 2': phrase2, 'Cosine Similarity': cosine_sim,
                                             'Euclidean Similarity': euclidean_sim})
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error on calculating '{phrase1}' and '{phrase2}' similarities: {e}")
                        return

        try:
            # Save as csv file
            similarities_df = pd.DataFrame(similarities)
            similarities_df.to_csv(output_file, index=False)
            print(f"results saved to file '{output_file}'")
        except Exception as e:
            print(f"Error on saving file to '{output_file}': {e}")

    def find_closest_match(self, user_input: str) -> Tuple[str, float]:
        """
        Find the closest match for a given user input phrase from a list of phrases.

        Args:
            user_input (str): The input phrase for which the closest match needs to be found.

        Returns:
            tuple: A tuple containing the closest matching phrase and its cosine similarity score.
        """
        max_cosine_similarity = -1
        closest_match = None

        for phrase in self.phrases_df['Phrases']:
            cosine_similarity, _ = self.get_similarity(user_input, phrase)
            if cosine_similarity > max_cosine_similarity:
                max_cosine_similarity = cosine_similarity
                closest_match = phrase

        return closest_match, max_cosine_similarity