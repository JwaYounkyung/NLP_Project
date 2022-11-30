import re
import unidecode
import spacy
import nltk
from autocorrect import Speller

nltk.download('words')

def context_preprocess(input_file):
    # open the given document
    with open(input_file, 'r') as f:
        text = f.read()
        
    
    # 1. remove the URLs
    text = remove_links(text)
    
    # 2. remove the words inside bracket
    text = remove_parentheses(text)
    
    # 3. remove the accented characters
    text = accented_characters_removal(text)
    
    # 4. remove the extra white spaces & extra newlines
    text = remove_newlines(text)
    text = remove_whitespace(text)
    
    # 5. Spelling Correction
    # text = spelling_correction(text)
    
    # 6. Split into Sentences
    # text = split_into_sentences(text)
    
    # 7. remove the non-english words
    # text = remove_non_english(text)
    
    # 8. Delete the titles of the document
    # text = delete_titles(text)

    output = text.split('\n')
    return output
       
    
    
# ref : https://towardsdatascience.com/cleaning-preprocessing-text-data-by-building-nlp-pipeline-853148add68a
def remove_links(text):
    remove_https = re.sub(r'http\S+', '', text)
    remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
    return remove_com

def accented_characters_removal(text):
    # Remove accented characters from text using unidecode.
    # Unidecode() - It takes unicode data & tries to represent it to ASCII characters.
    text = unidecode.unidecode(text)
    return text

def remove_whitespace(text):
    pattern = re.compile(r' +') 
    Without_whitespace = re.sub(pattern, ' ', text)
    # There are some instances where there is no space after '?' & ')', 
    # So I am replacing these with one space so that It will not consider two words as one token.
    text = Without_whitespace.replace('?', ' ? ').replace(')', ') ').replace(' , ', ', ')
    return text

def remove_newlines(text):
    pattern = re.compile(r'\n\t+')
    remove_newlines = re.sub(pattern, '\n', text)
    pattern = re.compile(r'\n\n+')
    remove_newlines = re.sub(pattern, '\n\n', remove_newlines)
    pattern = re.compile(r'\t+')
    remove_newlines = re.sub(pattern, '\t', remove_newlines)
    
    return remove_newlines


# ref : https://www.geeksforgeeks.org/how-to-remove-text-inside-brackets-in-python/
def remove_parentheses(text):
    new_text = re.sub("\(.*?\)", "", text)
    return new_text


def spelling_correction(text):   
    spell = Speller(lang='en')
    Corrected_text = spell(text)
    return Corrected_text

def remove_non_english(sentences):
    words = set(nltk.corpus.words.words())

    new_sentences = []
    for sent in sentences:
        new_sentences.append(" ".join(w for w in nltk.wordpunct_tokenize(sent) 
            if w.lower() in words or not w.isalpha()))
        
    return new_sentences

# ref : https://spacy.io/usage/linguistic-features#retokenization
def split_into_sentences(text):
    """
    input : preprocessed documents
    output : list of sentences in string format
    """
    nlp = spacy.blank('en')
    nlp.add_pipe('sentencizer')
    tokens = nlp(text)
    sentences = []
    for sent in tokens.sents:
        sentences.append(str(sent))

    return sentences

def delete_titles(text):
    new_sent = []
    for sent in text.split('\n'):
        if len(sent) > 0 and sent[-1] == '.':
            new_sent.append(sent)
    
    return '\n'.join(new_sent)
