CNN/DailyMail Summarizer

A Hugging Face Transformers–based abstractive summarization model fine-tuned on the CNN/DailyMail dataset.




About the generated output
- Each line in `generated_summaries.txt` corresponds to one test article  
- We evaluated 10 samples and stored them here for quick comparison  
- You can inspect diversity (Distinct-1, Distinct-2), coherence, and factual consistency across these examples

Model evaluation summary
1. Training & validation loss
   – Epochs: 5  
   – Final training loss: ~2.69 (down from ~3.6)  
   – Final validation loss: ~3.17 (no signs of overfitting)  
2. Structural fidelity (5/5)
   – Always uses “LOCATION (SOURCE) —” lead-in, context sentences, quotes, and reporter tags  
3. Lexical diversity (3/5)
   – Distinct-1 ~0.60 average (range 0.45–0.78)  
   – Distinct-2 ~0.80 average (range 0.75–0.88)  
4. Coherence (4/5)
   – Generally fluent, minor jumps occasionally  
5. Factual accuracy (2/5)
   – Hallucinates numbers/names when randomness is high; needs entity grounding

Recommendations
– Use decoding settings: `temperature=0.8`, `top_p=0.9`, `repetition_penalty=1.2`  
– Extract source entities (PERSON, GPE, CARDINAL) via spaCy and constrain or post-process outputs to match  
– Optionally fine-tune 1–2 more epochs at LR=1e-6 on a curated subset to reduce hallucinations  
– Perform human spot-checks on 50–100 summaries for factual consistency
