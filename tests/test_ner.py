from encite.ner import (
    _preprocess_text,
    _format_model_input,
    _parse_model_output,
    find_entities,
    Chunk,
    Entity,
)
from langchain_anthropic import ChatAnthropic
from unittest.mock import Mock


def test_preprocess_text():
    text = "Hello, World.\nTest"
    expected_chunks = [
        Chunk(text="Hello", start_char_index=0, end_char_index=5),
        Chunk(text="World", start_char_index=7, end_char_index=12),
        Chunk(text="Test", start_char_index=14, end_char_index=18),
    ]

    assert _preprocess_text(text) == expected_chunks


def test_preprocess_text_empty():
    text = ""
    assert _preprocess_text(text) == []


def test_preprocess_text_single_word():
    text = "Hello"
    expected_chunks = [
        Chunk(text="Hello", start_char_index=0, end_char_index=5),
    ]
    assert _preprocess_text(text) == expected_chunks


def test_preprocess_text_multiple_separators():
    text = "Hello,,,World...Test"
    expected_chunks = [
        Chunk(text="Hello", start_char_index=0, end_char_index=5),
        Chunk(text="World", start_char_index=8, end_char_index=13),
        Chunk(text="Test", start_char_index=16, end_char_index=20),
    ]
    assert _preprocess_text(text) == expected_chunks


def test_preprocess_text_all_separators():
    text = "  ,.\n\t  "
    assert _preprocess_text(text) == []


def test_preprocess_text_mixed_separators():
    text = "Hello\tWorld\n\nTest"
    expected_chunks = [
        Chunk(text="Hello", start_char_index=0, end_char_index=5),
        Chunk(text="World", start_char_index=6, end_char_index=11),
        Chunk(text="Test", start_char_index=13, end_char_index=17),
    ]
    assert _preprocess_text(text) == expected_chunks


def test_preprocess_text_trailing_word():
    text = "Hello, World, "
    expected_chunks = [
        Chunk(text="Hello", start_char_index=0, end_char_index=5),
        Chunk(text="World", start_char_index=7, end_char_index=12),
    ]
    assert _preprocess_text(text) == expected_chunks


def test_format_model_input():
    chunks = [
        Chunk(text="Hello", start_char_index=0, end_char_index=5),
        Chunk(text="World", start_char_index=6, end_char_index=11),
    ]
    entity_types = ["company"]
    system_prompt = "test prompt"
    expected_prompt = (
        system_prompt + "\nYou must extract the following entity types: ['company']"
    )

    expected_output = [
        {"role": "system", "content": expected_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "content",
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": "World"},
                        ],
                    },
                    "title": "Document",
                    "citations": {"enabled": True},
                }
            ],
        },
    ]

    assert _format_model_input(chunks, system_prompt, entity_types) == expected_output


def test_parse_model_output():
    chunks = [
        Chunk(text="Microsoft", start_char_index=0, end_char_index=9),
        Chunk(text="Corp", start_char_index=10, end_char_index=14),
    ]

    content_blocks = [
        {
            "type": "text",
            "text": "<entity_type>company</entity_type><name>Microsoft Corp</name>",
            "citations": [{"start_block_index": 0, "end_block_index": 2}],
        }
    ]

    expected_entities = [
        Entity(
            entity_type="company",
            name="Microsoft Corp",
            start_char_index=0,
            end_char_index=14,
        )
    ]

    assert _parse_model_output(content_blocks, chunks) == expected_entities


def test_parse_model_output_multiple_entities():
    chunks = [
        Chunk(text="John", start_char_index=0, end_char_index=4),
        Chunk(text="Smith", start_char_index=5, end_char_index=10),
        Chunk(text="works", start_char_index=11, end_char_index=16),
        Chunk(text="at", start_char_index=17, end_char_index=19),
        Chunk(text="Apple", start_char_index=20, end_char_index=25),
        Chunk(text="Inc", start_char_index=26, end_char_index=29),
    ]

    content_blocks = [
        {
            "type": "text",
            "text": "<entity_type>person</entity_type><name>John Smith</name>",
            "citations": [{"start_block_index": 0, "end_block_index": 2}],
        },
        {
            "type": "text",
            "text": "<entity_type>company</entity_type><name>Apple Inc</name>",
            "citations": [{"start_block_index": 4, "end_block_index": 6}],
        },
    ]

    expected_entities = [
        Entity(
            entity_type="person",
            name="John Smith",
            start_char_index=0,
            end_char_index=10,
        ),
        Entity(
            entity_type="company",
            name="Apple Inc",
            start_char_index=20,
            end_char_index=29,
        ),
    ]

    assert _parse_model_output(content_blocks, chunks) == expected_entities


def test_parse_model_output_no_entities():
    chunks = [
        Chunk(text="Hello", start_char_index=0, end_char_index=5),
        Chunk(text="World", start_char_index=6, end_char_index=11),
    ]

    content_blocks = [{"type": "text", "text": "No entities here", "citations": []}]

    assert _parse_model_output(content_blocks, chunks) == []


def test_parse_model_output_invalid_citation():
    chunks = [
        Chunk(text="Microsoft", start_char_index=0, end_char_index=9),
    ]

    # Missing citations field
    content_blocks = [
        {
            "type": "text",
            "text": "<entity_type>company</entity_type><name>Microsoft</name>",
        }
    ]

    assert _parse_model_output(content_blocks, chunks) == []


def test_parse_model_output_missing_tags():
    chunks = [
        Chunk(text="Microsoft", start_char_index=0, end_char_index=9),
    ]

    # Missing entity_type tag
    content_blocks = [
        {
            "type": "text",
            "text": "<name>Microsoft</name>",
            "citations": [{"start_block_index": 0, "end_block_index": 1}],
        }
    ]

    assert _parse_model_output(content_blocks, chunks) == []

    # Missing name tag
    content_blocks = [
        {
            "type": "text",
            "text": "<entity_type>company</entity_type>",
            "citations": [{"start_block_index": 0, "end_block_index": 1}],
        }
    ]

    assert _parse_model_output(content_blocks, chunks) == []


def test_find_entities():
    mock_model = Mock(spec=ChatAnthropic)
    mock_model.invoke.return_value.content = [
        {
            "type": "text",
            "text": "<entity_type>company</entity_type><name>Microsoft</name>",
            "citations": [{"start_block_index": 0, "end_block_index": 1}],
        }
    ]

    text = "Microsoft is a technology company."
    expected_entities = [
        Entity(
            entity_type="company",
            name="Microsoft",
            start_char_index=0,
            end_char_index=9,
        )
    ]
    system_prompt = "You're an NER expert."
    result = find_entities(
        mock_model, text, entity_types=["company"], system_prompt=system_prompt
    )
    expected_prompt = (
        system_prompt + "\nYou must extract the following entity types: ['company']"
    )

    assert result == expected_entities
    mock_model.invoke.assert_called_once()
    model_input = mock_model.invoke.call_args[0][0]
    assert model_input[0] == {"role": "system", "content": expected_prompt}
