---
title: PromptZip Backend
emoji: 🗜️
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# PromptZip Backend API

FastAPI backend for [PromptZip](https://github.com/SlothWriteCode/PromptZip) — compresses AI prompts using Microsoft's LLMLingua models.

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Check which models are loaded |
| POST | `/compress/fast` | LLMLingua-2 (fast, recommended) |
| POST | `/compress/standard` | LLMLingua original |
| POST | `/compress/long` | LongLLMLingua (for long documents) |

## Request Format

```json
{
  "prompt": "Your text here...",
  "ratio": 0.75,
  "preserve_structure": true
}
```

`ratio` = target compression strength. Higher values keep more of the original prompt.

`preserve_structure` = keeps lists, code fences, and instruction-heavy formatting safer during compression.

## Response Format

```json
{
  "compressed": "Compressed text...",
  "tokens_before": 42,
  "tokens_after": 21,
  "ratio_achieved": 0.5,
  "model_used": "LLMLingua-2"
}
```

## Deploy to Hugging Face Spaces

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Choose **Docker** as the SDK
3. Push this repo to that Space
4. Copy your Space URL (e.g. `https://your-username-promptzip-backend.hf.space`)
5. Paste it into PromptZip as the Backend URL

> **Note:** First startup takes ~60 seconds on free CPU tier while models download.
