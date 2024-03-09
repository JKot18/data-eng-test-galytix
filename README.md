# Word Embeddings CLI

This Command Line Interface (CLI) provides functionality to work with word embeddings, specifically utilizing a Word2Vec model for computing similarities between phrases.

## Installation

To use this CLI, ensure you have Python 3 installed on your system along with the necessary dependencies. You can install the dependencies by running:
```bash
pip install -r requirements.txt
```


## Usage

### Calculating Similarities

To calculate similarities between phrases and save the results to a file, use the following command:
```bash
python main.py --phrases-file <phrases_file_path> --embeddings-file <user_input> --user_input 
```


- `<phrases_file_path>`: Path to the file containing phrases.
- `<embeddings_file_path>`: Path to the binary file containing the Word2Vec model.
- `<user_input>`: Phrase to find the closest match for.

## Note

- Ensure that the phrases file (`phrases.csv` in this example) is in the correct format and encoding. Similarly, the embeddings file (`word2vec.bin` in this example) should be a valid binary file containing the Word2Vec model.
- The output file will be created in the 'data/similarities.csv' destination