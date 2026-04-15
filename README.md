# llamaindex-doubleword

LlamaIndex integration for [Doubleword](https://doubleword.ai) — real-time and batch inference.

## Install

```bash
pip install llamaindex-doubleword
```

## Usage

```python
from llamaindex_doubleword import DoublewordLLM

llm = DoublewordLLM(model="your-model")
response = llm.complete("Hello!")
```

See [docs.doubleword.ai](https://docs.doubleword.ai) for full documentation.
