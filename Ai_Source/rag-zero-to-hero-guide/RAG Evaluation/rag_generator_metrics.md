# RAG Generator Evaluation Metrics

Authored by [Kalyan KS](https://www.linkedin.com/in/kalyanksnlp/). To stay updated with LLMs, RAG and Agents, you can follow him on [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/), [Twitter](https://x.com/kalyan_kpl) and [YouTube](https://youtube.com/@kalyanksnlp?si=ZdoC0WPN9TmAOvKB).

## Faithfulness

The **Faithfulness metric** measures how factually consistent a generated response is with the retrieved context. It evaluates whether the claims  in the response can be inferred from the retrieved context.

The Faithfulness metric is computed based on:

- **Generated Response**: The answer provided by the RAG system.
- **Retrieved Context**: The text chunks retrieved by the RAG system.

The Faithfulness score is calculated as:

$$
\text{Faithfulness Score} = \frac{\text{Number of claims in the response supported by the retrieved context}}{\text{Total number of claims in the response}}
$$

---

**Metric Score Range**

- **Range**: 0 to 1
- **Interpretation**: A higher score indicates greater factual consistency with the retrieved context, while a lower score indicates discrepancies or unsupported claims.

**When Perfect and Worst Score Happens**

- **Perfect Score (1)**: Occurs when all claims in the generated response are fully supported by the retrieved context, with no contradictions.
- **Worst Score (0)**: Occurs when none of the claims in the generated response can be inferred from the retrieved context, indicating complete factual inconsistency.

---

**How This Metric Is Computed** 

1. **Extract Claims**: Use an LLM  to break down the generated response into individual factual claims.
2. **Verify Claims Against Context**: For each claim, check if it can be supported or inferred from the retrieved context using LLM or NLI models.
    - A claim is "supported" if it aligns with the context without contradiction.
    - A claim is "unsupported" if it contradicts the context or has no basis in it.
3. **Count Supported Claims**: Count the number of claims that are supported by the context.
4. **Calculate Score**: Compute faithfulness score as the ratio of number of  supported claims and the total number of claims.

---

**Example**

- **Question**: "When was the first Super Bowl held?"
- **Retrieved Context**: "The First AFL–NFL World Championship Game, later known as Super Bowl I, was played on January 15, 1967, at the Los Angeles Memorial Coliseum."
- **Generated Response**: "The first Super Bowl was held on January 15, 1967, in Florida."

**Step-by-Step Computation:**

1. **Extract Claims**:
    - Claim 1: "The first Super Bowl was held on January 15, 1967."
    - Claim 2: "The first Super Bowl was held in Florida."
    - Total claims = 2.
2. **Verify Claims Against Context**:
    - Claim 1: "The first Super Bowl was held on January 15, 1967."
        - Supported by context: "played on January 15, 1967."
        - Verdict: Yes (supported).
    - Claim 2: "The first Super Bowl was held in Florida."
        - Contradicted by context: "at the Los Angeles Memorial Coliseum" (in California, not Florida).
        - Verdict: No (unsupported).
3. **Count Supported Claims**:
    - Number of supported claims = 1.
4. **Calculate Score**:

$$
\text{Faithfulness Score} = \frac{\text{Number of supported claims}}{\text{Total claims}} = \frac{1}{2} = 0.5
$$

The response is only partially faithful. The date is correct, but the location contradicts the retrieved context, lowering the score.

## Hallucination

The hallucination metric evaluates how factually inconsistent is the generated response with the retrieved context.  It measures the extent to which the generated response contains fabricated or unsupported information not grounded in the context. Such fabricated or unsupported information is referred to as “hallucinations”. 

The hallucination metric is computed based on:

- **Generated Response**: The answer provided by the RAG system.
- **Retrieved Context**: The text chunks retrieved by the RAG system.

The hallucination metric is computed as 

$$
\text{Hallucination Score} = \frac{\text{Number of claims in the response unsupported by the retrieved context}}{\text{Total number of claims in the response}}
$$

---

**Metric Score Range**

- **Score Range**: 0 to 1
- **Interpretation**: A high score indicates significant factual inaccuracies  which low score indicates minimal factual inaccuracies.

**When Perfect and Worst Score Happens**

- **Perfect Score (1)**: occurs when the response is fully fabricated with all claims unsupported.
- **Worst Score (0)**: occurs when the response is fully consistent with the retrieved context, with no unsupported claims.

---

**How This Metric Is Computed** 

1. **Extract Claims**: Use an LLM  to break down the generated response into individual factual claims.
2. **Verify Claims Against Context**: For each claim, check if it can be supported or inferred from the retrieved context using LLM or NLI models.
    - A claim is "supported" if it aligns with the context without contradiction.
    - A claim is "unsupported" if it contradicts the context or has no basis in it.
3. **Count Supported Claims**: Count the number of claims that are not supported by the context.
4. **Calculate Score**: Compute hallucination score as the ratio of number of  unsupported claims and the total number of claims.

---

**Example**

- **Input**: "What is the capital of Brazil?"
- **Retrieval Context**: "Brazil is a country in South America. Its capital is Brasília."
- **Response**: "The capital of Brazil is Florida."
- **Steps**:
    1. **Claim Extraction**: The output contains one claim: "The capital of Brazil is Florida."
    2. **Claim Verification**: The retrieval context states the capital is Brasília, so the claim "The capital of Brazil is Florida" is unsupported and contradicts the context.
    3. **Count Supported Claims**: Number of supported claims = 1.
    4. **Compute Score**: Number of unsupported claims = 1, total claims = 1. Hallucination Score = .
        
        $\frac{1}{1} = 1$
        
- **Result**: Score = 1 (worst case, complete hallucination).

## Response Relevancy (LLM based)

The **Response Relevancy metric** evaluates how relevant a generated response is to the original user query. 

The Response Relevancy metric is computed based on:

- **User Query**: The original question asked by the user.
- **Generated Response**: The answer generated by the RAG system.

The Response Relevancy metric is  calculated as:

$$
\text{Response Relevancy Score} = \frac{\text{Number of relevant statements in the response}}{\text{Total number of statements in the response}}
$$

---

**Metric Score Range**

- **Range**: 0 to 1
- **Interpretation**: A score closer to 1 indicates high relevance to the user query, while a score closer to 0 indicates low relevance or off-topic content.

**When Perfect and Worst Score Happens**

- **Perfect Score (1)**: Occurs when the generated response fully and directly answers the query with no irrelevant details.
- **Worst Score (0)**: Occurs when the response is completely unrelated to the query, contains only irrelevant information.

---

**How this metric Is computed** 

1. **Statement Extraction**: Use an LLM to break the response into individual statements.
2. **Relevance Assessment**: Use an LLM to evaluate each statement’s relevance to the user query. The LLM directly scores each statement as relevant (1) or irrelevant (0) based on semantic alignment.
3. **Count Relevant Statements**: Count the number of relevant statements in the response. 
4. **Score Calculation**: Compute the ratio of relevant statements to total statements in response.

---

**Example**

- **Query**: "What is the capital city of Brazil?"
- **Generated Response**: "The capital city of Brazil is Brasília. It replaced Rio de Janeiro as the capital in 1960. Florida is a state in the USA."

**Step-by-Step Computation:**

1. **Statement Extraction**:
    - Statement 1: "The capital city of Brazil is Brasília."
    - Statement 2: "It replaced Rio de Janeiro as the capital in 1960."
    - Statement 3: "Florida is a state in the USA."
    - Total statements = 3.
2. **Relevance Assessment**:
    - Statement 1: Directly answers the query (Relevant = 1).
    - Statement 2: Provides additional context about the capital, still relevant to the query (Relevant = 1).
    - Statement 3: Unrelated to the query about Brazil’s capital (Irrelevant = 0).
3. **Count Relevant Statements**: Number of relevant statements = 2
4. **Score Calculation**:

$$
\text{Response Relevancy Score} = \frac{\text{Number of relevant statements}}{\text{Total statements}} = \frac{2}{3} \approx 0.67
$$

The response is mostly relevant, correctly identifying Brasília as the capital and providing related context, but the mention of Florida introduces irrelevant information, reducing the score.

## Response Relevancy (String based)

The **Response Relevancy** metric assesses how relevant a generated response is to the user query. It evaluates whether the response directly addresses the user query without focusing on factual correctness. 

The Response Relevancy metric is computed based on:

- **User Query**: The original question asked by the user.
- **Generated Response**: The answer generated by the RAG system.

The Answer Relevancy is computed as the mean cosine similarity between the embedding of the user query and the embeddings of the synthetic queries generated from the response:

$$
\text{Response Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \cos(E_{g_i}, E_o)

$$

$$
\text{Response Relevancy} = \frac{1}{N} \sum_{i=1}^{N} \frac{E_{g_i} \cdot E_o}{\|E_{g_i}\| \|E_o\|}
$$

Where:

- $E_o$ : Embedding of the user query.
- $E_{g_i}$ : Embedding of the (i)-th synthetic query.
- (N): Number of synthetic queries (default is 3).
- $\cos(E_{g_i}, E_o)$ : Cosine similarity between the embeddings.

Note: While the score typically ranges from 0 to 1 in practice, cosine similarity can theoretically range from -1 to 1, though negative values are rare in this context.

---

**Metric Score Range**

- **Range**: Typically between 0 and 1 in practice, though mathematically, it can range from -1 to 1 due to the nature of cosine similarity.
- **Interpretation**: Higher scores (closer to 1) indicate better relevance, while lower scores (closer to 0 or negative) indicate poor relevance.

**When Perfect and Worst Score Happens**

- **Perfect Score (1)**: Occurs when the generated answer is relevant to the original question.
- **Worst Score (0)**: Occurs when the answer is irrelevant, incomplete, or contains excessive unrelated information.

---

**How This Metric Is Computed (Steps)**

1. **Generate Synthetic Queries**: Use an LLM to generate (N) synthetic queries (default ) based solely on the generated response. 
    - Example: For the answer "The first Super Bowl was held on Jan 15, 1967," the LLM might generate:
        - "When did the first Super Bowl take place?"
        - "What was the date of the initial Super Bowl?"
        - "On which day was the first Super Bowl held?"
2. **Generate Embeddings**: Convert the user query and each generated query into vector embeddings using a text embedding model.
3. **Calculate Cosine Similarity**: Compute the cosine similarity between the embedding of the user query and the embedding of each generated query.
4. **Average the Scores**: Take the mean of the (N) cosine similarity scores to obtain the final Response Relevancy score.

Here is the underlying idea is that if the answer is highly relevant, the generated questions will closely resemble the original question, resulting in high cosine similarity.

---

**Example**

- **Question**: "When was the first Super Bowl?"
- **Answer**: "The first Super Bowl was held on Jan 15, 1967."

**Step-by-Step Computation:**

1. **Generated Questions** (by LLM based on the answer):
    - Q1: "When did the first Super Bowl occur?"
    - Q2: "What was the date of the first Super Bowl?"
    - Q3: "On which day was the first Super Bowl held?"
2. **Embeddings**: Assume embeddings are generated (simplified for illustration):
    - $E_o$  (original question embedding): [0.9, 0.4, 0.1]
    - $E_{g1}$:   [0.85, 0.38, 0.09]
    - $E_{g2}$: [0.88, 0.39, 0.11]
    - $E_{g3}$: [0.87, 0.41, 0.10]
3. **Cosine Similarity** (hypothetical values):
    - $\cos(E_{g1}, E_o) = 0.98$
    - $\cos(E_{g2}, E_o) = 0.99$
    - $\cos(E_{g3}, E_o) = 0.97$
4. **Mean Calculation**:
    - $\text{Answer Relevancy} = \frac{0.98 + 0.99 + 0.97}{3} = 0.98$

A score of  0.98  indicates the answer is highly relevant to the question).

**Counter Example:**

- **Question**: "When was the first Super Bowl?"
- **Answer**: "Football is a popular sport."
- **Generated Questions**:
    - Q1: "What is a popular sport?"
    - Q2: "Why do people like football?"
    - Q3: "What makes football famous?"
- **Cosine Similarity**: Likely low (e.g., 0.2, 0.1, 0.15) due to dissimilarity with "When was the first Super Bowl?"
- **Score**:  (low, indicating irrelevance).
    
    $$
    \frac{0.2 + 0.1 + 0.15}{3} = 0.15
    $$
    

## Relevant Noise Sensitivity

Relevant Noise Sensitivity refers to the ratio of incorrect claims in a model’s response that are entailed by relevant retrieved chunks to the total number of response claims. A relevant chunk is a piece of retrieved context that contains at least one claim from the ground truth answer. 

This metric evaluates how sensitive the RAG system’s generator is to the noise (misleading or incorrect information) present alongside useful information in the retrieved context. A high value indicates that the generator frequently produces incorrect responses by relying on noisy data from relevant chunks.

This metric is computed based on 

- **Ground Truth Answer**: The correct reference answer to the user query.
- **Retrieved Relevant Chunks**: Chunks that contain at least one claim matching the ground truth.
- **Model Response**: The generated output from the RAG system.

The Relevant Noise Sensitivity (RNS) is computed using the formula:

$$
\text{Relevant Noise Sensitivity} = \frac{\text{Number of incorrect claims in the model response entailed by relevant chunks}}{\text{Total number of claims in the response}}
$$

---

**Metric Score Range**

- **Range**: 0 to 1
- **Interpretation**: Higher score means the generator is highly sensitive to noise while lower score means the generator is more robust to noise.

---

**When Perfect and Worst Score Happens**

- **Perfect Score (0)**: Occurs when the model response contains no incorrect claims, or when any incorrect claims present are not supported by any relevant retrieved chunks. This indicates the generator successfully filters out noise from relevant context.
- **Worst Score (1)**: Occurs when every incorrect claim in the model response is directly supported by a relevant retrieved chunk. This suggests that the generator is entirely misled by noise in the relevant context.

---

**How this metric is computed (step by step)**

1. **Identify Ground Truth Claims**: Break down the ground truth answer into individual factual claims
2. **Identify Relevant Chunks**: From the retrieved context, identify relevant chunks i.e., chunks with contain at least one claim from the ground truth 
3. **Extract Model Response Claims**: Break the model’s response into individual claims 
4. **Classify Claims**: Compare model response claims to the ground truth to label them as correct or incorrect.
5. **Check Entailment**: For each incorrect claim in the response, determine if it is entailed (supported or implied) by a relevant chunk 
6. **Count and Compute**:
    - Numerator: Count incorrect claims entailed by relevant chunks.
    - Denominator: Count total claims in the response.
    - Divide to get the relevant noise sensitivity score.

---

**Example**

- **User Query**: "Who painted the Mona Lisa and in what century was it painted?"
- **Ground Truth Answer**: "Leonardo da Vinci painted the Mona Lisa in the 16th century."
- **Retrieved Relevant Chunk**: "The Mona Lisa is a famous portrait painted by Leonardo da Vinci. It is believed to have been started in the early 1500s. Some art historians date its completion to around 1519, placing it firmly in the 15th century according to certain periodizations."
- **Model Response**: "Leonardo da Vinci painted the Mona Lisa, and it was painted in the 15th century."

**Step by Step Computation**

1. **Ground Truth Claims**:
    - Claim 1: "Leonardo da Vinci painted the Mona Lisa" .
    - Claim 2: "It was painted in the 16th century" .
2. **Relevant Chunk Identification**: The retrieved chunk is relevant because it contains "Leonardo da Vinci painted the Mona Lisa," matching the ground truth.
3. **Model Response Claims**:
    - Claim 1: "Leonardo da Vinci painted the Mona Lisa" .
    - Claim 2: "It was painted in the 15th century".
4. **Classify Claims**:
    - Correct: 1 claim ("Leonardo da Vinci painted the Mona Lisa").
    - Incorrect: 1 claim ("It was painted in the 15th century").
5. **Check Entailment**:
    - Incorrect claim ("15th century") is entailed by the relevant chunk’s statement: "...placing it firmly in the 15th century according to certain periodizations."
6. **Compute RNS**:
    - Numerator: 1 (incorrect claim entailed by a relevant chunk).
    - Denominator: 2 (total claims in the response).
    - $RNS = \frac{1}{2} = 0.5$

The Relevant Noise Sensitivity is **0.5**, indicating that half of the claims in the model response are incorrect and stem from noise in a relevant retrieved chunk. This reflects moderate sensitivity to noise.

## Irrelevant Noise Sensitivity

Irrelevant Noise Sensitivity (INS) refers to  the ratio of incorrect claims in a model's response that are entailed by irrelevant retrieved chunks to the total number of response claims . An irrelevant chunk is a retrieved piece of context that does not contain any claims from the ground truth answer. A high INS value indicates that the model’s generator is overly influenced by noise from irrelevant context, leading it to include incorrect  information in its response.

**This metric is computed based on**

- **Ground Truth Answer**: The correct answer, used to define valid claims.
- **Model Response**: The RAG system’s output, from which claims are extracted.
- **Irrelevant Retrieved Chunks**: Chunks that contain no ground truth claims.

The INS is calculated as:

$$
\text{Irrelevant Noise Sensitivity} = \frac{\text{Number of incorrect claims in the response entailed by irrelevant chunks}}{\text{Total number of claims in the response}}
$$

---

**Metric Score Range**

Metrics Score Range: 0 to 1  

Interpretation:  High Score indicates the model is highly sensitive to irrelevant noise, meaning most incorrect claims in the response are derived from irrelevant retrieved chunks. Low Score suggests the model is robust against irrelevant noise, with few or no incorrect claims supported by irrelevant chunks.

**When Perfect and Worst Scores Happen**

- **Perfect Score (0)**: The response either has no incorrect claims, or any incorrect claims are not derived from irrelevant chunks (e.g., they might come from the model’s own errors or relevant chunks).
- **Worst Score (1)**: Every incorrect claim in the response is traceable to irrelevant retrieved chunks, indicating the model is highly susceptible to irrelevant noise.

---

**How This Metric Is Computed (Steps)**

1. **Extract Ground Truth Claims**: Identify the claims in the ground truth answer to the query.
2. **Extract Model Response Claims**: Break down the model’s response into individual claims.
3. **Identify Irrelevant Chunks**: Determine which chunks are irrelevant (do not entail any ground truth claim).
4. **Identify Incorrect Claims**: From the model’s response, find claims that do not match the ground truth.
5. **Check Entailment with Irrelevant Chunks**: For each incorrect claim, determine if it is supported by any irrelevant chunk.
6. **Calculate the Ratio**:
    - Numerator: Count incorrect claims entailed by irrelevant chunks.
    - Denominator: Count all claims in the model’s response.
    - Divide to get the INS.

---

**Example**

**Query**: "Who wrote the novel 'Pride and Prejudice'?" 

**Ground Truth Answer**: "Jane Austen wrote 'Pride and Prejudice.'"

**Model Response**: "Charlotte Brontë wrote 'Pride and Prejudice,' and she is famous for 'Jane Eyre.'"

**Retrieved Chunks**:

- "Jane Austen published 'Pride and Prejudice' in 1813, a classic romance novel.".
- "Charlotte Brontë, a renowned author, is best known for her novel 'Jane Eyre,' published in 1847."

**Steps**:

1. **Identify Ground Truth Claims**: GT1 = "Jane Austen wrote 'Pride and Prejudice.'"
2. **Identify Model Claims**: M1 = "Charlotte Brontë wrote 'Pride and Prejudice,'" M2 = "Charlotte Brontë is famous for 'Jane Eyre.'"
3. **Identify Irrelevant Chunks**:  IR1- "Charlotte Brontë, a renowned author, is best known for her novel 'Jane Eyre,' published in 1847." is irrelevant.
4. **Identify Incorrect Claims**:
    - M1 (incorrect).
    - M2 (incorrect).
5. **Entailment Check**:
    - M1 ("Charlotte Brontë wrote 'Pride and Prejudice'") is not entailed by IR1.
    - M2 ("Charlotte Brontë is famous for 'Jane Eyre'") is entailed by IR1.
6. **Calculation**:
    - Numerator: Number of incorrect claims entailed by irrelevant chunks = 1 (only M2).
    - Denominator: Total number of claims in the response = 2 (M1 and M2).
    - INS =  $\frac{1}{2} = 0.5$

50% of the model’s claims that deviate from the ground truth are supported by irrelevant retrieved chunks. This suggests the model is moderately sensitive to irrelevant noise.