# Social Network Analysis of Novels

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddeepak95/social-network-analysis-novels/blob/main/social-network-analysis-novels-colab.ipynb)

## Overview

This project explores the application of Social Network Analysis (SNA) to literary novels. By extracting and analyzing the relationships between characters, we can gain new insights into narrative structure, character importance, and the dynamics of fictional worlds.

## How is the social network constructed?

The core of this project is the automated extraction of character interactions from the text of a novel. Here’s how the process works:

1. **Character Name Extraction:**  
   The notebook uses natural language processing (NLP) techniques to identify and extract the names of characters from the novel. This may involve:

   - Named Entity Recognition (NER) to detect person names.
   - Manual or semi-automated curation to handle aliases, nicknames, or variations in character names.

2. **Interaction Detection:**  
   Character interactions are inferred based on their co-occurrence within a defined window of text (such as a sentence, paragraph, or fixed number of words). If two or more character names appear within the same window, an interaction (edge) is recorded between them.

3. **Network Construction:**

   - Each unique character is represented as a node in the network.
   - An edge is drawn between two characters if they interact (i.e., co-occur in the same window).
   - The weight of each edge can represent the frequency of interactions (how often the two characters appear together).

4. **Network Analysis:**  
   Once the network is built, various network analysis techniques (such as centrality measures and community detection) are applied to gain insights into the structure and dynamics of the story.

**Note:**  
The window size for co-occurrence and the method for name extraction can be customized in the notebook, allowing for flexible analysis depending on the novel and research question.

## Getting Started

You can run the analysis directly in your browser using Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ddeepak95/social-network-analysis-novels/blob/main/social-network-analysis-novels-colab.ipynb)

### Requirements

- Python 3.x
- Jupyter Notebook or Google Colab
- Libraries: NetworkX, matplotlib, pandas, (and any others used in your notebook)

All dependencies are pre-installed in Colab.

### Running Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/ddeepak95/social-network-analysis-novels.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook in your desired IDE.
