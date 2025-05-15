# RAG Retriever Evaluation Metrics

Authored by [Kalyan KS](https://www.linkedin.com/in/kalyanksnlp/). To stay updated with LLMs, RAG and Agents, you can follow him on [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/), [Twitter](https://x.com/kalyan_kpl) and [YouTube](https://youtube.com/@kalyanksnlp?si=ZdoC0WPN9TmAOvKB).

## Context Precision

Context Precision is a metric that evaluates how well a retriever ranks  relevant chunks within the retrieved context. It assesses whether the relevant chunks appear at higher ranks, reflecting the RAG system's ability to prioritize useful information over irrelevant or noisy data.

Context Precision metric  is computed based on: 

- **Reference (Ground Truth)**: The ideal or correct answer to the query, serving as the standard.
- **Retrieved Context**: The chunks retrieved by the RAG system to address the query.

The Context Precision at K (denoted as Context Precision@K) is computed as follows:

$$
\text{Context Precision@K} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{\text{Total number of relevant chunks in the top } K \text{ chunks}}
$$

Where:

- ( K ): The total number of chunks in the retrieved contexts.
- $v_k$: A relevance indicator at rank ( k ), where  $v_k = 1$  if the chunk is relevant, and $v_k = 0$ if it is not.
- $\text{Precision@k} = \frac{\text{true positives@k}}{\text{true positives@k} + \text{false positives@k}}$: The precision at rank ( k ), calculated as the ratio of relevant chunks up to position ( k ) to the total number of chunks up to ( k ).
- The denominator normalizes the score by the total number of relevant chunks  in the top ( K ) chunks.

---

**Metric score range**

- **Score range**: 0 to 1
- **Interpretation**: A high score  occurs when  relevant chunks are ranked at the top of the list. A low score occurs when  relevant chunks are buried lower in the ranking, overshadowed by irrelevant ones.

**When perfect and worst score happens**

- **When perfect score happens**: A perfect score of 1 is achieved when all retrieved chunks are relevant to the ground truth, and they are ranked in an order that perfectly prioritizes them (i.e., no irrelevant chunks precede relevant ones).
- **When worst score happens**: A score of 0 is obtained when none of the retrieved chunks are relevant to the ground truth, regardless of their ranking.

---

**How this metric is computed (Steps)**

1. **Identify Relevance**: For each chunk in the retrieved contexts, determine if it is relevant to the reference answer. 
2. **Assign Relevance Indicators**: Assign $v_k = 1$  for relevant chunks and  $v_k = 0$ for irrelevant ones.
3. **Calculate Precision@k**: For each rank ( k ) (from 1 to  K), compute the precision as the number of relevant chunks up to that rank divided by the total number of chunks up to that rank.
4. **Compute Weighted Sum**: Multiply each $\text{Precision@k}$  by its corresponding $v_k$  and sum these values across all ranks.
5. **Compute Score**: Divide the sum by the total number of relevant chunks in the top ( K ) chunks.

---

**Example**

- **Question**: "What is the largest desert in the world?"
- **Ground Truth**: "The largest desert in the world is the Antarctic Desert, which spans about 14 million square kilometers."
- **Retrieved Contexts**:
    1. "The Antarctic Desert is the largest desert by area, covering 14 million square kilometers."
    2. "The Sahara Desert is a large desert in Africa."
    3. "Deserts are dry regions with little rainfall."

**Step by step computation**

1. **Relevance Check**:
    - Chunk 1: Relevant  – directly answers the query by identifying the largest desert.
    - Chunk 2: Irrelevant – mentions a desert but not the largest one.
    - Chunk 3: Irrelevant  – too general, doesn’t specify the largest desert.
2. **Assign Relevance Indicators**:

    - Chunk 1: Relevant ( $v_1 = 1$) 
    - Chunk 2: Irrelevant ( $v_2 = 0$) 
    - Chunk 3: Irrelevant ($v_3 = 0$) 

1. **Compute Precision@k**:
    - At $k = 1$ : 1 relevant / 1 total = 1.0
    - At  $k = 2$: 1 relevant / 2 total = 0.5
    - At $k = 3$ : 1 relevant / 3 total = 0.33
2. **Compute Weighted Sum**:
    - $(1.0 \times 1) + (0.5 \times 0) + (0.33 \times 0) = 1.0 + 0 + 0 = 1.0$
3. **Compute Score**: Total relevant items in ground truth = 1 (only the Antarctic Desert is correct).
    - $\text{Context Precision@3} = \frac{1.0}{1} = 1.0$

Context Precision = 1.0  because the only relevant chunk is ranked at the top.

Now, if the order was reversed (irrelevant chunks ranked higher), the score would drop, reflecting poorer precision due to lower ranking of the relevant chunk.

## Context Recall

Context recall evaluates the completeness of the retrieved context in a RAG pipeline. It assesses whether the retriever successfully fetches all the relevant pieces of information required to answer the query. It is computed as the ratio of number of ground truth claims found in retrieved context to the total number of ground truth claims. 

Context Recall metric  is computed based on: 

- **Reference (Ground Truth)**: The ideal or correct answer to the query, serving as the standard.
- **Retrieved Contexts**: The information or documents retrieved by the RAG system to address the query.

The formula for Context Recall (LLM-based) is:

$$
\text{Context Recall} = \frac{\text{Number of ground truth claims found in retrieved context}}{\text{Total number of claims in ground truth}}
$$

---

**Metric Score Range**

- **Score Range**: 0 to 1
- **Interpretation**: A high score happens when the retrieved context contains most or all of the information required to answer the query, indicating strong retrieval.  A low score happens when the retrieved context lacks significant portions of the information required to answer the query, indicating weak retrieval.

**When Perfect and worst score happens** 

- **When Perfect Score Happens (1)**: All claims or pieces of information in the reference answer are fully supported by the retrieved context.
- **When Worst Score Happens (0)**: None of the claims in the reference answer can be linked to the retrieved context, showing a total retrieval failure.

---

**How this metric is computed** 

1. **Extract ground truth claims**: The LLM breaks down the reference (ground truth) answer into individual claims or statements, where each claim represents a unique piece of information.
2. **Analyze retrieved context**: For each claim in the reference, the LLM checks if the retrieved context contains enough information to substantiate it.
3. **Count attributable claims**:  Count the number of attributable ground truth claims. 
4. **Compute the score** : Divide the number of attributable claims by the total number of claims in the reference answer to produce a score between 0 and 1.

---

**Example**

- **User Input**: "What are the primary causes of deforestation?"
- **Reference (Ground Truth)**: "The primary causes of deforestation are logging, agriculture, urbanization, and wildfires."
- **Retrieved Contexts**: ["Logging is a major driver of deforestation worldwide.", "Agriculture and urban development contribute significantly to forest loss."]

**Step by step computation:**

1. **Break Down Reference into Claims**:
    - Claim 1: "Logging is a cause of deforestation"
    - Claim 2: "Agriculture is a cause of deforestation"
    - Claim 3: "Urbanization is a cause of deforestation"
    - Claim 4: "Wildfires are a cause of deforestation"
2. **Analyze Retrieved Context**:
    - Claim 1 : Supported by "Logging is a major driver of deforestation worldwide."
    - Claim 2 : Supported by "Agriculture and urban development contribute significantly to forest loss."
    - Claim 3 : Supported by "Agriculture and urban development contribute significantly to forest loss."
    - Claim 4 : Not mentioned in the retrieved context.
3. **Count Attributable Claims**: 3 out of 4 claims are supported.
4. **Calculate Context Recall**:

$$
\text{Context Recall} = \frac{3}{4} = 0.75
$$

The Context Recall score is 0.75, meaning 75% of the information in the reference answer is present in the retrieved context. The score falls short of 1 because "wildfires" as a cause of deforestation is absent from the retrieved context.

## Context Entities Recall

Context Entities Recall assesses how good the RAG system’s retriever is in fetching the entities in the reference answer. It is computed as the ratio of number of common entities between reference answer and the retrieved context to the total number of entities in the reference answer. 

This metric is computed based on:

- **Reference Entities (RE)**: Entities extracted from the reference (ground truth) answer.
- **Retrieved Context Entities (RCE)**: Entities extracted from the retrieved context.

The Context Entities Recall is calculated as follows:

$$
\text{Context Entities Recall} = \frac{\text{Number of common entities between RCE and RE}}{\text{Total number of entities in RE}}
$$

or symbolically:

$$
\text{Context Entities Recall} = \frac{|RCE \cap RE|}{|RE|}
$$

- $|RCE \cap RE|$: The number of entities present in both the retrieved context and the reference answer.
- (|RE|): The total number of entities in the reference answer.

---

**Metric Score Range**

- **Score Range**: 0 to 1
- **Interpretation**: A high score happens when when most or all of the entities in the reference are present in the retrieved context while a low score happens when few of the entities in the reference are found in the retrieved context.

**When perfect and worst score happens**

- **When perfect score happens**: A perfect score of 1 is achieved when all entities in the reference are present in the retrieved context (100% recall of entities).
- **When worst score happens**: A score of 0 occurs when none of the entities in the reference are found in the retrieved context (0% recall of entities).

---

**How this metric is computed (Steps)**

1. **Extract Entities from Reference**: Identify and list all distinct entities (e.g., names, locations, dates) in the reference answer.
2. **Extract Entities from Retrieved Context**: Identify and list all distinct entities in the retrieved context provided by the RAG system.
3. **Find Common Entities**: Determine the intersection of the two sets, i.e., the entities that appear in both the reference and the retrieved context.
4. **Compute score**: Divide the number of common entities by the total number of entities in the reference to compute the recall score.

---

**Example**

Let’s consider a question: "What is the capital of Brazil, and when was its current capital established?"

- **Reference (Ground Truth)**: "The capital of Brazil is Brasília, established on April 21, 1960."
- **Retrieved Context**: "Brasília is a city in Brazil, designed as the capital."

**Step-by-Step Computation**:

1. Extract reference entities: ["Brazil", "Brasília", "April 21, 1960"]
2. Extract retrieved context entities: ["Brasília", "Brazil"]
3. Find common entities: ["Brazil", "Brasília"] (Intersection: 2 entities)
4. Compute score: $\text{Context Entities Recall} = \frac{2}{3} = 0.6667$

The Context Entities Recall score is approximately 0.67, indicating that 67% of the entities in the reference were recalled by the retrieved context. The date "April 21, 1960" was missing, lowering the score from a perfect 1.

## Context Relevancy

Context Relevancy metric evaluates how relevant is the retrieved context to the user’s query. It is computed as the ration of number of statements in the context relevant to the user’s query to the total number of statements in the retrieved context. 

Context Recall metric  is computed based on: 

- **Query** : The original question asked by the user.
- **Retrieved Contexts**: The information or documents retrieved by the RAG system to address the query.

The context relevancy score is calculated as:

$$
\text{Contextual Relevancy} = \frac{\text{Number of statements in the context relevant to the user’s query}}{\text{Total number of Statements in the retrieved context}}
$$

---

**Metric score range**

- **Score Range**: 0 to 1
- **Interpretation**: A high score happens when most of the statements in the retrieved context are relevant to the user query. A low score happens when only a few of the statements in the retrieved context are relevant to the user query.

**When perfect and worst score happens**

- **When perfect score happens**: A perfect score of 1 is achieved when every statement in the retrieval context is relevant to the input, with no irrelevant content retrieved.
- **When worst score happens**: A worst score of 0 is assigned when none of the statements in the retrieval context are relevant to the input, meaning the retriever failed entirely to fetch useful information.

---

**How this metric is computed (Steps)**

1. **Statement Extraction**: An LLM extracts all individual statements from the retrieval context. A statement is a distinct piece of information or claim within the retrieved text.
2. **Relevance Assessment**: The same LLM evaluates each extracted statement to determine if it is relevant to the input query. Relevance is assessed based on whether the statement contributes meaningfully to addressing the query.
3. **Compute Score**: The number of relevant statements is divided by the total number of statements in the retrieval context to compute the score. This ratio reflects the proportion of relevant content retrieved.

---

**Example**

- **Input**: "What are the benefits of drinking green tea?"
- **Retrieval Context**:
    - "Green tea contains antioxidants that may reduce the risk of chronic diseases."
    - "Coffee is a popular beverage worldwide."
    - "Green tea can improve brain function due to its caffeine content."

**Steps by step computation**:

1. **Statement Extraction**:
    - Statement 1: "Green tea contains antioxidants that may reduce the risk of chronic diseases."
    - Statement 2: "Coffee is a popular beverage worldwide."
    - Statement 3: "Green tea can improve brain function due to its caffeine content."
2. **Relevance Assessment**:
    - Statement 1: Relevant (directly addresses a benefit of green tea).
    - Statement 2: Irrelevant (unrelated to green tea benefits).
    - Statement 3: Relevant (directly addresses a benefit of green tea).
3. **Compute score** :
    - Number of Relevant Statements = 2
    - Total Number of Statements = 3
    - Contextual Relevancy = $\frac{2}{3}$ ≈ 0.67

A score of 0.67, indicates moderate context relevancy, as two-thirds of the retrieved content is useful for answering the query.