# corellm — LLM integration library

`corellm` is an LLM integration library that provides an easy-to-use API for running local large language models (LLM) (via `llama-cpp-python`) and for instantly building a web GUI (via Gradio). The library exposes a single high-level `LLM` class for chat/prompt interactions (including streaming) and an `Interface` class to render a Gradio-based chat interface.

Local LLMs can be downloaded as `.gguf` models. You can find these on [Hugging Face](https://huggingface.co/models?library=gguf).

**`corellm`** is currently **in development**

---

# Quick Start

```bash
pip install corellm
```

```python
from corellm import LLM

llm = LLM(path="models/your-model.gguf")
resp = llm.chat("Hello — what can you do?")
print(resp)
```

Note: `corellm` uses `llama-cpp-python` under the hood to load and run GGUF models. Gradio is used only by the optional `Interface` class. These dependencies are automatically installed when downloading `corellm`.

---

# Concepts

* **Memory**: `LLM` keeps a conversation history in `memory`, a list of role/content dictionaries (`{"role": "system"|"user"|"assistant", "content": str}`) similar to standard chat formats. The first element is always the system prompt.
* **System prompt**: The initial system instruction (defaults to `"You are a helpful CoreLLM assistant."`). It is stored as `memory[0]`.
* **Stateless prompt**: A one-off prompt that does not modify `memory`.
* **Streaming**: Token-by-token output is available via generators for both chat and stateless prompts.

---

# API Reference

> The API below documents classes and methods you can interact with. Types and behavior are written to match the exposed runtime behavior.

## `class corellm.LLM`

Create an LLM instance and interact with the model.

### `LLM(path: str, system_prompt: str = "You are a helpful CoreLLM assistant.", max_tokens: int = 4096) -> None`

Create a new LLM instance.

* `path`: Filesystem path to the GGUF model file to load.
* `system_prompt`: Initial system message persisted at the start of `memory`.
* `max_tokens`: Maximum context window (passed to the underlying model).

Raises: Errors from the model loader (for example if `path` is invalid).

Example:

```python
llm = LLM(path="models/your-model.gguf")
```

### `chat(prompt: str) -> str`

Send a prompt as a user message and store both the prompt and the assistant response in conversation memory.

* `prompt`: Text to send to the model.
* Returns: the assistant's response string.

Behavior:

* Appends `{"role":"user","content": prompt}` to memory before calling the model.
* Appends `{"role":"assistant","content": response}` to memory after receiving the reply.

Example:

```python
resp = llm.chat("Summarize this paragraph.")
print(resp)
# memory now contains the user prompt and the assistant response
```

### `prompt(prompt: str, use_memory: bool = True) -> str`

Send a prompt but **do not** modify persistent conversation memory.

* `prompt`: Text to send to the model.
* `use_memory`: If `True`, the model call uses the current memory as context (so it is influenced by prior messages); if `False`, only the current system prompt and the supplied prompt are used.
* Returns: the assistant's response string.

Example:

```python
# Ask a one-off question without recording it in memory
answer = llm.prompt("Translate to Spanish: 'Good morning'", use_memory=False)
```

### `chat_stream(prompt: str, print_stream: bool = False) -> Generator[str] | None`

Stream tokens for a `chat()`-style interaction and add the final response to memory.

* `prompt`: The prompt text to send (will be appended to memory as a user message).
* `print_stream`: If `True`, tokens are printed to stdout as they are produced and the method **auto-runs** the stream and returns `None`. If `False`, the method returns a generator that yields tokens (strings) which you can iterate to consume the streamed output.
* Returns: a generator that yields token chunks (if `print_stream` is `False`), or `None` if `print_stream` is `True`.

Behavior notes:

* The full assistant message is appended to memory after the stream completes.
* If you want to collect the whole response while streaming, iterate the generator and concatenate tokens.

Examples:

```python
# Manual consumption of stream
gen = llm.chat_stream("Explain recursion simply.")
response = ""
for token in gen:
    response += token
print("Full response:", response)

# Auto-run printing
llm.chat_stream("Tell me a short joke.", print_stream=True)
# prints the tokens as they arrive and returns None
```

### `prompt_stream(prompt: str, print_stream: bool = False, use_memory: bool = True) -> Generator[str] | None`

Stream tokens for a stateless prompt (does not add assistant response to memory).

* `prompt`: The prompt text.
* `print_stream`: If `True`, tokens are printed to stdout while the stream runs and the method returns `None`. If `False`, it returns a generator you can iterate.
* `use_memory`: If `True`, the stream uses the current memory as context; if `False`, only system prompt + the prompt are used.
* Returns: a generator yielding token chunks (if `print_stream` is `False`), otherwise `None`.

Example:

```python
# Collect tokens from the stream and build the final string
tokens = []
for t in llm.prompt_stream("Write a haiku about autumn.", use_memory=False):
    tokens.append(t)
haiku = "".join(tokens)
print(haiku)
```

### `clear_memory() -> None`

Reset conversation memory to only the system prompt.

Example:

```python
llm.clear_memory()
```

### `set_memory(memory: list[dict[str, str]]) -> None`

Replace the entire conversation memory.

* `memory` must be a list of dictionaries where each dict has at least the keys `"role"` and `"content"`. Typical roles: `"system"`, `"user"`, `"assistant"`.

Example:

```python
llm.set_memory([
    {"role": "system", "content": "You are an assistant that replies in bullet points."},
    {"role": "user", "content": "List the steps to boil an egg."},
    {"role": "assistant", "content": "1. ... 2. ..."}
])
```

### `get_memory() -> list[dict[str, str]]`

Return the current conversation memory.

Example:

```python
history = llm.get_memory()
```

### `modify_system_prompt(system_prompt: str) -> None`

Change the system prompt (overwrites memory[0]).

Example:

```python
llm.modify_system_prompt("You are concise and formal.")
```

---

## `class corellm.Interface`

A small utility class that builds and launches a Gradio-based chat interface.

### `Interface(llm: LLM, port: int = 3001) -> None`

Create an `Interface` bound to an `LLM` instance.

* `llm`: A `corellm.LLM` instance.
* `port`: Port to run the Gradio server on.

### `render() -> None`

Launch the Gradio chat interface in a browser. The UI will reflect the LLM's current memory (user/assistant pairs are shown). The interface streams responses if the underlying LLM streams.

Notes:

* This requires Gradio to be installed (`gradio` package).
* The function launches a local server and opens a browser tab by default (Gradio standard behavior).

Example:

```python
from corellm import LLM, Interface

llm = LLM(path="models/your-model.gguf")
ui = Interface(llm, port=3001)
ui.render()
```

---

# Examples

## 1) Simple chat (stateful)

```python
from corellm import LLM

llm = LLM(path="models/your-model.gguf")
print(llm.chat("Hello! Who are you?"))

# The model remembers that conversation
print(llm.chat("What did I ask you previously?"))
```

## 2) One-off stateless prompt

```python
summary = llm.prompt("Summarize this text: <paste text here>", use_memory=False)
print(summary)

# memory is unchanged by prompt(..., use_memory=False)
```

## 3) Streaming and collecting tokens manually

```python
gen = llm.chat_stream("Explain the difference between TCP and UDP.")
collected = []
for token in gen:
    collected.append(token)
final_response = "".join(collected)
print(final_response)
# full response is now also appended to memory
```

## 4) Streaming with automatic printing

```python
# prints tokens to stdout as they arrive; returns None
llm.prompt_stream("Write a short limerick about a coder.", print_stream=True, use_memory=False)
```

## 5) Memory manipulation

```python
# Reset to fresh system prompt
llm.clear_memory()

# Overwrite memory with a custom conversation
llm.set_memory([
    {"role":"system","content":"You are an assistant that always answers in JSON."},
    {"role":"user","content":"Provide metadata for a book."}
])

# Inspect
print(llm.get_memory())

# Change system prompt (does not remove other messages)
llm.modify_system_prompt("You are an assistant that replies in YAML.")
```

## 6) Launching the GUI (optional)

```python
from corellm import LLM, Interface

llm = LLM(path="models/your-model.gguf")
ui = Interface(llm, port=3001)
ui.render()  # launches Gradio and opens browser
```

---

# Usage notes & best practices

* **Model path**: Provide the correct GGUF model file path when constructing `LLM`. The library will pass this to `llama_cpp` to load the model.
* **Dependencies**: `llama-cpp-python` is used for low-level operations with LLM models. `gradio` is used in the `Interface` class to render the chat window.
* **Memory size and tokens**: Be mindful of the `max_tokens`/context window. Large histories can exceed the model context limit and may cause errors or truncated context.
* **Streaming behavior**:

  * If you pass `print_stream=True` to `chat_stream` or `prompt_stream`, the method will print tokens as they arrive and will return `None`. If you need the generator to process tokens yourself, set `print_stream=False`.
  * The `chat_stream` method appends the final assistant message to memory after the stream completes; `prompt_stream` does not.
* **Threading / concurrency**: The underlying model runtime may not be thread-safe. For concurrent usage, create separate `LLM` instances per thread or process, or serialize model access.
* **Error handling**: Errors from model loading or runtime will bubble up from `llama-cpp-python` (e.g., file not found, out-of-memory). Catch exceptions around model creation and calls as needed.

---

# Troubleshooting

* **Model fails to load**: Confirm the `path` you provided is correct and points to a GGUF model file compatible with your `llama-cpp` build.
* **No output or empty responses**: Check model compatibility and the tokenization/context length. Try with a smaller prompt or a different model to isolate the issue.
* **Streaming does not yield**: If you call streaming with `print_stream=True`, the function will auto-consume the stream and return `None`. To get a generator you can iterate, use `print_stream=False`.
* **GUI does not open**: If `Interface.render()` seems to hang, check your environment (headless servers often need `inbrowser=False` or `share=True` depending on your configuration). Also ensure `gradio` is installed.

---

# Minimal requirements

* Python 3.8+
* Optional runtime dependencies only if you use those features:

  * `llama-cpp-python` — required for model execution
  * `gradio` — required only to use `Interface`

---

# Design intent

`corellm` aims to provide a clear and compact API for integrating local LLMs into applications with minimal friction:

* Make it straightforward to do stateful chat and stateless prompts.
* Provide token-level streaming so applications can render progressive output.
* Keep a simple and inspectable conversation memory model.
* Offer a small, optional web UI for development and demos.

`corellm`'s API exposes a stable `LLM` and `Interface` abstraction so you can integrate model-powered features fast.

---

# Contributing

Contributions, bug reports, and feature requests are welcome. When opening issues or PRs, include:

* Minimal reproduction (snippets)
* Model path or example model (if relevant)
* Error tracebacks

Updates to `corellm` are done by Jadon Duff.

---

# License

**MIT License**

Copyright (c) 2025 Jadon Duff

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
