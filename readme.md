## KDSH 2026 – Story Consistency (Track A + Track B)

### Approach
We verify character backstories against full novel texts using
long-context retrieval and semantic Natural Language Inference (NLI).

Each backstory is split into atomic claims.
Relevant novel passages are retrieved via embeddings and evaluated using
an ensemble of NLI models to detect entailment or contradiction.

Final decisions are made by aggregating claim-level verdicts.
Thresholds are calibrated automatically using the training set.

### Output
prediction = 1 → consistent  
prediction = 0 → contradict

Track B explanations are provided per test example.
