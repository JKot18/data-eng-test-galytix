from src.word_embeddings import WordEmbeddings as we

import argparse
import os


def main():
    """
       Command-line interface for finding the closest match for a user-input phrase from a list of phrases.

       This function parses command-line arguments to specify the paths to the CSV file containing phrases,
       the Word2Vec binary file, and the user-input phrase. It then validates the existence of the specified
       files, creates a WordEmbeddings object, and finds the closest match for the user-input phrase.
       """
    parser = argparse.ArgumentParser(description='Find the closest match for a user-input phrase from a list of phrases.')
    parser.add_argument('phrases_file', type=str, help='Path to the CSV file containing phrases.')
    parser.add_argument('embeddings_file', type=str, help='Path to the Word2Vec binary file.')
    parser.add_argument('user_input', type=str, help="Phrase to find the closest match for")
    args = parser.parse_args()

    # Validate phrases file
    if not os.path.isfile(args.phrases_file):
        print(f"Error: Phrases file '{args.phrases_file}' not found.")
        return


    # Validate embeddings file
    if not os.path.isfile(args.embeddings_file):
        print(f"Error: Embeddings file '{args.embeddings_file}' not found.")
        return

    try:
        word_embeddings = we(args.embeddings_file, args.phrases_file)
        closest_match, similarity = word_embeddings.find_closest_match(args.user_input)
        print(f"Closest match to '{args.user_input}': '{closest_match}', Cosine Similarity: {similarity}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()