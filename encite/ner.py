import re
from typing import cast
from dataclasses import dataclass
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic


ENTITY_TYPE_RE = re.compile(r"<entity_type>(.*?)</entity_type>", re.DOTALL)
ENTITY_NAME_RE = re.compile(r"<name>(.*?)</name>", re.DOTALL)
SEPARATORS = {" ", ".", ",", "\n", "\t"}


NER_SYSTEM_PROMPT = """You are a named entity recognition (NER) expert.
Extract all named entities from the provided document.

INSTRUCTIONS:

- make sure to include citations
- each citation must be a single entity
- each citation must include the entity type and name in the text, e.g. "<entity_type>company</entity_type><name>Microsoft</name>"
"""


class Entity(BaseModel):
    """Entity name from Claude citation."""

    entity_type: str
    name: str
    start_char_index: int  # absolute character position in original text
    end_char_index: int  # absolute character position in original text


@dataclass
class Chunk:
    """Represents a chunk of text with its character offsets."""

    text: str
    start_char_index: int
    end_char_index: int

    def to_dict(self) -> dict:
        """Convert to format expected by Claude."""
        return {"type": "text", "text": self.text}


def _preprocess_text(text: str) -> list[Chunk]:
    """Split text into chunks based on separators, tracking character positions.

    Args:
        text: Input text to chunk

    Returns:
        List of Chunk objects containing text and character positions
    """
    chunks = []
    current_chunk = []
    chunk_start = 0

    for i, char in enumerate(text):
        if char in SEPARATORS:
            if current_chunk:  # Only create chunk if we have accumulated characters
                chunk_text = "".join(current_chunk)
                chunks.append(
                    Chunk(
                        text=chunk_text, start_char_index=chunk_start, end_char_index=i
                    )
                )
                current_chunk = []
        else:
            if not current_chunk:  # Start of new chunk
                chunk_start = i
            current_chunk.append(char)

    # Handle last chunk if exists
    if current_chunk:
        chunk_text = "".join(current_chunk)
        chunks.append(
            Chunk(
                text=chunk_text, start_char_index=chunk_start, end_char_index=len(text)
            )
        )

    return chunks


def _format_model_input(
    chunks: list[Chunk], system_prompt: str, entity_types: list[str]
) -> list[dict]:
    prompt = (
        system_prompt + f"\nYou must extract the following entity types: {entity_types}"
    )
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "content",
                        "content": [c.to_dict() for c in chunks],
                    },
                    "title": "Document",
                    "citations": {"enabled": True},
                }
            ],
        },
    ]


def _parse_model_output(
    content_blocks: list[dict], chunks: list[Chunk]
) -> list[Entity]:
    entities: list[Entity] = []

    for block in content_blocks:
        if (
            block.get("type") != "text"
            or not isinstance(block.get("citations"), list)
            or not block["citations"]
        ):
            continue

        text = block.get("text", "")
        entity_type_match = ENTITY_TYPE_RE.search(text)
        name_match = ENTITY_NAME_RE.search(text)

        if not entity_type_match or not name_match:
            continue

        entity_type = entity_type_match.group(1)
        name = name_match.group(1)

        citation = block["citations"][0]
        start_block = cast(int, citation["start_block_index"])
        end_block = cast(int, citation["end_block_index"])

        start_char_index = chunks[start_block].start_char_index
        end_char_index = chunks[end_block - 1].end_char_index
        entities.append(
            Entity(
                entity_type=entity_type,
                name=name,
                start_char_index=start_char_index,
                end_char_index=end_char_index,
            )
        )

    return entities


def find_entities(
    model: ChatAnthropic,
    text: str,
    *,
    entity_types: list[str],
    system_prompt: str = NER_SYSTEM_PROMPT,
) -> list[Entity]:
    """Find named entities in text using Claude's citations feature.

    Args:
        model: ChatAnthropic model instance to use for entity recognition
        text: Input text to analyze
        entity_types: List of entity types to look for (e.g., ["person", "company"])
        system_prompt: Custom system prompt to use (defaults to NER_SYSTEM_PROMPT)

    Returns:
        List of Entity objects containing the found entities with their types,
        names and character positions in the original text
    """
    chunks = _preprocess_text(text)
    model_input = _format_model_input(chunks, system_prompt, entity_types)
    response = model.invoke(model_input)
    entities = _parse_model_output(response.content, chunks)
    return entities
