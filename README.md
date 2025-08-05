CNN/DailyMail Summarizer

A Hugging Face Transformers–based abstractive summarization model fine-tuned on the CNN/DailyMail dataset.

Script Overview:

`finetunedmodel.py` does the following:

1. **Environment check**  
   - Detects and reports GPU availability and memory.

2. **Dataset loading & truncation**  
   - Reads `/root/cnn_dailymail_full.txt`.  
   - Truncates to 50 MB if larger.  
   - Tokenizes with GPT-2 tokenizer (block size 128).  
   - Creates training examples (up to 10 000).

3. **Train/validation split**  
   - 90% of examples for training, 10% for validation.

4. **Model initialization**  
   - Loads `gpt2` model and tokenizer.  
   - Resizes token embeddings and moves model to GPU if available.

5. **TrainingArguments optimized for Vast.ai**  
   - `num_train_epochs=5`  
   - `per_device_train_batch_size=4` (with gradient accumulation → effective BS=8)  
   - `learning_rate=5e-5`, `warmup_steps=100`, `weight_decay=0.01`  
   - Mixed precision (`fp16=True`), frequent logging & checkpointing  
   - Early stopping on best `eval_loss`

6. **Training and checkpointing**  
   - Saves best model to `/root/creative-finetuned-model`.  
   - Logs to `/root/creative_logs`.  
   - Final validation loss printed.

7. **Text generation tests**  
   - Uses Hugging Face `pipeline("text-generation")` on 10 CNN-style prompts.  
   - Prints samples for three creativity settings (temperature=1.2, 0.8, 0.5).  
   - Tips for news-style writing in comments.

Outputs
-------
• **Model checkpoint**: `/root/creative-finetuned-model/`  
• **Training logs**: `/root/creative_logs/`  
• **Generated samples**: printed to console during script run  

Usage
-----
1. Upload your dataset file to `/root/cnn_dailymail_full.txt` on Vast.ai.  
2. SSH into the instance and run:





About the generated output
- Each line in `generated output.txt` corresponds to one test article  
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
