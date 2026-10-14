# MatBERT batch encoding audit

The active embedding implementation tokenizes a batch to `[B,L]`, passes the full tokenizer dictionary including `attention_mask` into `model(**inputs)`, and extracts `outputs.last_hidden_state[:, 0, :]`, producing `[B,768]`. No batch-first-item replication, `repeat`, `expand`, or batch-axis mean is used. `audit_10task.py` additionally records ten tokenized examples per task and verifies more than one unique attended token sequence.
