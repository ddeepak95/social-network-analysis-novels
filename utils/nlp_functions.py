import json
from utils.epub_functions import html_to_markdown
import spacy
from collections import Counter
import re
import string
import math
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from itertools import combinations

def extract_character_names_with_variations(book_content: list, output_loc: str, start_chapter_index: int = 1, end_chapter_index: int = None, clean_names: bool = True) -> None:
    """
    Extract character names from book content and save variations for manual editing.
    
    Args:
        book_content: List of chapter dictionaries containing book content
        output_loc: Path to save the output JSON file
        start_chapter_index: First chapter to process (inclusive)
        end_chapter_index: Last chapter to process (exclusive). If None, processes all chapters.
        clean_names: Whether to perform basic cleaning of punctuations and merge character names automatically. Defaults to True.
    """
    print("Note: This process can take significant time for large books. Please be patient.")
    print("Loading spaCy transformer model... This may take a moment.")
    # Load spaCy transformer model
    nlp = spacy.load("en_core_web_trf")
    print("Model loaded successfully.")

    # Extract all PERSON entities
    all_names = []
    # If end_chapter_index is None, process all chapters from start_chapter_index onwards
    chapters_to_process = book_content[start_chapter_index:] if end_chapter_index is None else book_content[start_chapter_index:end_chapter_index+1]
    
    print(f"Processing {len(chapters_to_process)} chapters... This may take some time depending on the book size.")
    for i, chapter in enumerate(chapters_to_process, start=1):
        print(f"Processing chapter {i} of {len(chapters_to_process)}...")
        chapter_content = html_to_markdown(chapter["content"])
        doc = nlp(chapter_content)
        all_names.extend([ent.text for ent in doc.ents if ent.label_ == "PERSON"])
        print(f"Found {len([ent for ent in doc.ents if ent.label_ == 'PERSON'])} names in this chapter.")
    print("Processing complete. Analyzing results...")
    # Use all unique names (no frequency cutoff)
    unique_names = sorted(set(all_names))

    # Prepare a dict for manual editing: canonical name -> [variations]
    character_variations = {name: [name] for name in unique_names}

    if clean_names:
        print("Cleaning and merging character names...")
        def normalize_apostrophe(s):
            return s.replace("’", "'")

        def clean_key(key):
            key = normalize_apostrophe(key)
            key = key.strip()
            # Remove trailing punctuation (period, comma, etc.) and whitespace
            key = re.sub(rf"[{re.escape(string.punctuation)}\s]+$", "", key)
            # Remove possessive 's
            key = re.sub(r"'s$", "", key)
            return key.strip()

        # Merge variations
        merged = {}
        for orig_key, variations in character_variations.items():
            base_key = clean_key(orig_key)
            # Find the original key in the dict that matches the cleaned key, if any
            canonical_key = next((k for k in merged if clean_key(k) == base_key), base_key)
            merged.setdefault(canonical_key, set())
            merged[canonical_key].update(variations)
            merged[canonical_key].add(orig_key)  # Add the original key as a variation

        # Convert sets back to lists and sort
        character_variations = {k: sorted(list(v)) for k, v in merged.items()}
        print(f"Cleaned names from {len(unique_names)} to {len(character_variations)} unique characters.")

    # Save to JSON for manual editing
    with open(output_loc, "w", encoding="utf-8") as f:
        json.dump(character_variations, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(character_variations)} character candidates to '{output_loc}'.\n\n Take a moment and Edit this file to add variations for each character. This is important for correct social network creation.")


def character_counter(full_text, character_variations, output_loc):
    # Count mentions for each character group
    character_counter = Counter()
    for canonical, variations in character_variations.items():
        count = 0
        for name in variations:
            pattern = r'\b' + re.escape(name) + r'\b'
            count += len(re.findall(pattern, full_text, re.IGNORECASE))
            character_counter[canonical] = count
    
    # Sort characters by count descending
    sorted_counts = dict(sorted(character_counter.items(), key=lambda x: x[1], reverse=True))
    
    # Print top 5 characters and counts
    print("Top 5 characters by mention count:")
    for character, count in list(sorted_counts.items())[:5]:
        print(f"{character}: {count} mentions")
    
    # save the sorted character_counter to a json file
    with open(output_loc, "w", encoding="utf-8") as f:
        json.dump(sorted_counts, f, ensure_ascii=False, indent=2)

    print(f"Character counter saved to {output_loc}")




def plot_character_wordcloud(char_counts, scale_type='linear', top_n=20):
    """
    Display a wordcloud from character counts.

    Args:
        char_counts (dict): Dictionary of character counts {character: count}.
        scale_type (str): 'linear' or 'log' scaling for frequencies.
        top_n (int): Number of top characters to include.
    """
    # Sort and select top_n
    sorted_counts = dict(sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])

    # Apply scaling
    if scale_type == 'log':
        scaled_counts = {k: (math.log(v+1) if v > 0 else 0) for k, v in sorted_counts.items()}
    else:  # linear
        scaled_counts = sorted_counts

    # Generate wordcloud
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(scaled_counts)

    print(f"Wordcloud generated for {len(scaled_counts)} characters. Scale type: {scale_type}")

    # Display
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def bar_plot_character_counts(char_counts, top_n=20):
    # Sort characters by count descending
    sorted_counts = dict(sorted(char_counts.items(), key=lambda x: x[1], reverse=True)[:top_n])

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(sorted_counts.keys(), sorted_counts.values())
    plt.xlabel('Character')
    plt.ylabel('Count')
    plt.title(f'Top {top_n} Characters by Mention Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def extract_character_interactions(
    character_variations_path,
    book_content,
    output_path,
    granularity="line"
):
    """
    Extracts character interactions from book content and saves to a JSON file.

    Args:
        character_variations_path (str): Path to character variations JSON.
        book_content (str): The full book text as a single string.
        output_path (str): Path to save the output JSON.
        granularity (str): 'line' (default) or 'para' for processing.
    """
    print("Loading character variations...")
    try:
        with open(character_variations_path, "r", encoding="utf-8") as f:
            character_variations = json.load(f)
    except Exception as e:
        print(f"Error loading character variations: {e}")
        return

    print("Preparing variation mapping...")
    variation_to_canonical = {}
    for canonical, variations in character_variations.items():
        for v in variations:
            variation_to_canonical[v.lower()] = canonical

    print(f"Splitting text into {'paragraphs' if granularity == 'para' else 'lines'}...")
    if granularity == "para":
        units = [p.strip() for p in re.split(r'\n\s*\n', book_content) if p.strip()]
    else:  # default: line
        units = [line.strip() for line in re.split(r'\n+', book_content) if line.strip()]

    if not units and book_content:
        units = [book_content]
    elif not units:
        print("Warning: No text content found.")
        return

    print("Preparing regex pattern for character variations...")
    all_variations = sorted(variation_to_canonical.keys(), key=len, reverse=True)
    variation_pattern = r'\b(' + '|'.join(map(re.escape, all_variations)) + r')\b' if all_variations else None

    print("Analyzing character co-occurrences...")
    pair_cooccurrence_counter = Counter()

    if variation_pattern:
        for idx, unit in enumerate(units):
            canonical_chars = set()
            for match in re.finditer(variation_pattern, unit, re.IGNORECASE):
                variation = match.group(1).lower()
                if variation in variation_to_canonical:
                    canonical_chars.add(variation_to_canonical[variation])
            if len(canonical_chars) >= 2:
                for char1, char2 in combinations(sorted(canonical_chars), 2):
                    pair = (char1, char2)
                    pair_cooccurrence_counter[pair] += 1

    print("Preparing results...")
    interaction_data = []
    for pair, count in pair_cooccurrence_counter.items():
        interaction_data.append({
            "character_1": pair[0],
            "character_2": pair[1],
            "cooccurrences": count
        })

    if interaction_data:
        print("Sorting and saving results...")
        interaction_data = sorted(
            interaction_data,
            key=lambda x: (-x['cooccurrences'], x['character_1'], x['character_2'])
        )
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(interaction_data, f, ensure_ascii=False, indent=4)
            print(f"Data saved to {output_path} in sorted order")
        except Exception as e:
            print(f"Save error: {e}")
    else:
        print("No co-occurrences found.")

