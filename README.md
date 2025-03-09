# Encite

Encite is an experimental Python library for named entity recognition (NER) using Anthropic Claude models with citations feature.

## Features

- 🎯 Accurate entity recognition using Anthropic Claude's citations feature
- ⚙️ Customizable entity types (person, company, location, etc.)
- 📍 Returns entity character locations in the original text

## Installation

```bash
pip install encite
```

## Usage

> [!TIP]
> Claude 3.7 Sonnet is recommended for best results.

Here's a simple example of how to use Encite:

```bash
export ANTHROPIC_API_KEY=your-api-key
```

```python
from encite import find_entities
from langchain_anthropic import ChatAnthropic

# Initialize the Claude model
model = ChatAnthropic(model="claude-3-7-sonnet-latest")

# Example text
text = "John Smith works at Apple Inc. in California."

# Extract entities
entities = find_entities(model, text, entity_types=["person", "company"])

# Print results
print(entities)
```

Expected output:

```
[
    Entity(entity_type="person", name="John Smith", start_char_index=0, end_char_index=10),
    Entity(entity_type="company", name="Apple Inc", start_char_index=20, end_char_index=29)
]
```