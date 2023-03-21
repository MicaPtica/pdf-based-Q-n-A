import streamlit as st
import PyPDF2
import io
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import tiktoken
import openai
import pandas as pd
import numpy as np
import sklearn
import os

def remove_newlines(serie):
    serie = serie.replace('\n', ' ')
    serie = serie.replace('\\n', ' ')
    serie = serie.replace('  ', ' ')
    serie = serie.replace('  ', ' ')
    return serie

def convert_pdf_to_txt_pages(path):
    texts = ''
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)

    interpreter = PDFPageInterpreter(rsrcmgr, device)
    size = 0
    c = 0
    file_pages = PDFPage.get_pages(path)
    nbPages = len(list(file_pages))
    for page in PDFPage.get_pages(path):
      interpreter.process_page(page)
      t = retstr.getvalue()
      if c == 0:
        texts+=t
      else:
        texts+=t
      c = c+1
      size = len(t)
    # text = retstr.getvalue()

    # fp.close()
    device.close()
    retstr.close()
    return texts, nbPages




# Load the cl100k_base tokenizer which is designed to work with the ada-002 model

# @st.cache
pdf_file = st.file_uploader("Load your PDF", type="pdf")

if pdf_file:
    path = pdf_file.read()
    text,_ = convert_pdf_to_txt_pages(pdf_file)
    st.write(text[:10]+'.......')
    
    # if False:

    tokenizer = tiktoken.get_encoding("cl100k_base")
    max_tokens = 100

    # Function to split the text into chunks of a maximum number of tokens
    def split_into_many(text, max_tokens = max_tokens):

        # Split the text into sentences
        sentences = text.split('. ')

        # Get the number of tokens for each sentence
        n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
        
        chunks = []
        tokens_so_far = 0
        chunk = []

        # Loop through the sentences and tokens joined together in a tuple
        for sentence, token in zip(sentences, n_tokens):

            # If the number of tokens so far plus the number of tokens in the current sentence is greater 
            # than the max number of tokens, then add the chunk to the list of chunks and reset
            # the chunk and tokens so far
            if tokens_so_far + token > max_tokens:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0

            # If the number of tokens in the current sentence is greater than the max number of 
            # tokens, go to the next sentence
            if token > max_tokens:
                continue

            # Otherwise, add the sentence to the chunk and add the number of tokens to the total
            chunk.append(sentence)
            tokens_so_far += token + 1

        return chunks
        
    @st.cache_data 
    def get_emb(text):
    # # Loop through the dataframe
    # for row in df.iterrows():

    #     # If the text is None, go to the next row
    #     if row[1]['text'] is None:
    #         continue

    #     # If the number of tokens is greater than the max number of tokens, split the text into chunks
    #     if row[1]['n_tokens'] > max_tokens:
        shortened = split_into_many( remove_newlines(str(text)))

        df = pd.DataFrame(shortened, columns = ['text'])
        df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

        openai.api_key = os.environ["OPENAI_API_KEY"]

        df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    # df.to_csv('processed/embeddings UG4_8_GL_FAQ.csv')
    # df.head()
        df['embeddings'] = df['embeddings'].apply(np.array)
        st.write(df)
        return df
    
    df=get_emb(text)

    from openai.embeddings_utils import distances_from_embeddings, cosine_similarity

    # df=pd.read_csv('processed\embeddings UG4_8_GL_FAQ.csv', index_col=0)
    # df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    st.info('Embeddings created')

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():
        
        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4
        
        # If the context is too long, break
        if cur_len > max_len:
            break
        
        # Else add it to the text that is being returned
        returns.append(row["text"])

    # Return the context
    return "\n\n###\n\n".join(returns)

def answer_question(
    df,
    model="text-davinci-003",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )
    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create a completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""
    


text=st.text_input('Ask your question:')
if text:
    st.write(answer_question(df, question=text))