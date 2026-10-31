# Practicing Four LLM Prompting Techniques

**Student:** SAN Piseth
**Workshop:** Dr. Bouy Rina, Techpreneur 2.0
**Notebook:** [`LLM_Prompting_Assignment.ipynb`](LLM_Prompting_Assignment.ipynb)

This notebook works through four prompting techniques using a small local LLM. It is adapted from [Dr. Rina's original notebook](https://github.com/rinabuoy/llm-fundamentals-tp2/blob/main/notebooks%20-%20session%202/LLM_Prompting_Ollama_Colab_TP.ipynb).

## Model

- **`gemma4:e2b`**, served locally through [Ollama](https://ollama.com).
- Called with the `ollama` Python client, with `think=False`. Deterministic runs use `temperature=0.0`. Sampled runs (self-consistency) use `temperature=0.8`.
- Retrieval uses scikit-learn TF-IDF with cosine similarity. It does not use embeddings.

## Runtime

- Google Colab with a **GPU runtime**.
- The setup cell installs `ollama`, `scikit-learn`, and `numpy`, plus the Ollama server through `apt`/`curl`. It then starts `ollama serve` and pulls the model.
- The setup commands are Linux shell commands, so the notebook is meant to run in Colab, not directly on Windows.
- Run the cells in order. Downloading the model on first setup takes a few minutes.

## Exercises

| # | Technique | Task | Observed result |
|---|-----------|------|-----------------|
| 1 | Zero-shot vs. few-shot | Extract `name`, `city`, and `age` from a sentence as JSON | Zero-shot wrapped the JSON in Markdown fences, so strict `json.loads` failed. With two examples, the output was valid JSON with the correct values. |
| 2 | Chain-of-thought | Bat-and-ball puzzle (answer: $0.05) | The step-by-step prompt reached $0.05. The direct prompt started working through algebra and hit the 60-token limit before giving an answer. |
| 3 | Self-consistency | Implement `self_consistent(prompt, k)` and compare k=1 with k=7 at temperature 0.8 | k=1 gave no answer the parser could read. At k=7, 2 valid votes were for 0.05 and 5 answers could not be parsed, so the winner was 0.05. |
| 4 | RAG | Add two fictional Acme Corp facts, rebuild the TF-IDF index, and compare answers with and without retrieval | With retrieval, the model answered "17 percent" and "$2.75 per order". It said "I don't know" for the CEO question, which none of the documents answers. Without retrieval, it declined to answer all three questions. |

The notebook also has four short demos (zero/few-shot, CoT, self-consistency, RAG) taken from the instructor's notebook.

## Limitations

- **Small samples.** Each comparison is one prompt or a handful of samples on one small model. The results show how the techniques work. They do not measure accuracy.
- **Answer parsing is fragile.** In Exercise 3, 5 of 7 samples ended with `Answer:` and no plain number after it, or used LaTeX, so the regex could not read them. They were left out of the vote, not counted as wrong. The vote therefore rests on only 2 valid answers.
- **Token limits change outcomes.** The direct prompt in Exercise 2 was cut off by `max_tokens=60`, so it did not get a fair chance to answer.
- **TF-IDF retrieval is basic.** It matches on shared words, not meaning. In the Demo 4 run (k=1), the model replied that the context did not say how many days a customer has to return an item, even though the refund policy states 45 days. Either the wrong document was retrieved, or the model missed the match between "return" and "returned". For the CEO question, related but irrelevant documents were retrieved.
- **The facts are fictional.** The Acme Corp documents are made-up exercise data.
- **Results are not reproducible run to run.** Sampled outputs at temperature 0.8 vary between runs. Model or Ollama updates can also change the deterministic outputs.
- **Not included:** the optional challenge and the OCR fine-tuning demo.
