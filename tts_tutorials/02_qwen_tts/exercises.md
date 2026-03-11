# Exercises

## 1. Dual-Track Inference Trade-offs

Qwen-TTS utilizes a "Dual-Track Architecture" featuring a Semantic Track (25Hz) and a Latency Track (12.5Hz). 
In which scenario would you force the model to use the Semantic Track? In which scenario would the Latency Track be necessary? Explain your reasoning in terms of latency vs. audio fidelity.

## 2. Multi-Token Prediction (MTP) Bottleneck

Why is predicting all 16 audio codec layers sequentially for every time step a severe bottleneck for Large Language Models? How does the Multi-Token Prediction (MTP) module alleviate this computational burden?

## 3. The Power of Text Prompts

Explain how a token-based, LLM-style architecture like Qwen-TTS inherently supports capabilities like "prompt-based voice design" (e.g., typing "a gravelly, middle-aged male voice") compared to traditional parameter-based TTS systems.