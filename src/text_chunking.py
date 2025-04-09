import tiktoken
from typing import List, Optional


def chunk_text(text: str, max_tokens: int = 1000, delimiters: Optional[List[str]] = None) -> List[str]:
    """
    Split text into chunks of approximately max_tokens, ensuring chunks end at natural delimiters.

    Args:
        text: The text to be chunked
        max_tokens: The target maximum number of tokens per chunk (default: 1000)
        delimiters: Optional list of delimiter characters to split on. If None, defaults to ['.', '!', '?', '\n']

    Returns:
        A list of text chunks, each ending with a natural delimiter and approximately max_tokens in length
    """
    if not text:
        return []

    if delimiters is None:
        delimiters = ['.', '!', '?', '\n']

    # Use tiktoken for accurate tokenization
    encoding = tiktoken.encoding_for_model("gpt-4o")  # Using OpenAI's encoding

    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0

    # Split text into sentences first
    sentences = []
    current_sentence = ""

    for char in text:
        current_sentence += char
        if char in delimiters:
            sentences.append(current_sentence)
            current_sentence = ""

    # Add the last sentence if it's not empty
    if current_sentence:
        sentences.append(current_sentence)

    # Group sentences into chunks based on token count
    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))

        if current_chunk_tokens + sentence_tokens <= max_tokens:
            current_chunk += sentence
            current_chunk_tokens += sentence_tokens
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
            current_chunk_tokens = sentence_tokens

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)

    return chunks
