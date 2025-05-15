# RAG Evaluation Metrics

Authored by [Kalyan KS](https://www.linkedin.com/in/kalyanksnlp/). To stay updated with LLMs, RAG and Agents, you can follow him on [LinkedIn](https://www.linkedin.com/in/kalyanksnlp/), [Twitter](https://x.com/kalyan_kpl) and [YouTube](https://youtube.com/@kalyanksnlp?si=ZdoC0WPN9TmAOvKB).

## Introduction

RAG helps to reduce hallucinations in LLM generated responses by retrieving relevant context from external knowledge sources.  RAG consists of two core components: the retriever and the generator. Retriever fetches the information relevant to the user query from external knowledge sources and Generator generates the desired answer based on the query and retrieved information. 

<p align="center">
    <img src="images/RAG_Evaluation_Metrics.gif" alt="What is RAG" width="600" height="300">
</p>


## Evaluating RAG Retriever

The retriever based on the input query, fetches the relevant information from external knowledge sources. This ensures that generator (LLM) has access to  up-to-date and pertinent data.  

Evaluating the retriever is essential because its performance directly impacts the quality of the generated output.  If  retriever fails to retrieve relevant or complete context, the LLM may produce inaccurate or irrelevant answers.

Some of the popular metrics for RAG retriever evaluation are context precision, context recall, context entities recall and context relevancy.

<table border="1">
  <tr>
    <th>Metric</th>
    <th>Inputs</th>
    <th>Type</th>
    <th>Assesses</th>
  </tr>
  <tr>
    <td>Context Precision</td>
    <td>Context, Reference</td>
    <td>Reference Dependent</td>
    <td>Retriever’s ability to rank relevant chunks higher in the retrieved context</td>
  </tr>
  <tr>
    <td>Context Recall</td>
    <td>Context, Reference</td>
    <td>Reference Dependent</td>
    <td>Retriever’s ability to fetch the relevant pieces of information to answer the user query.</td>
  </tr>
  <tr>
    <td>Context Entities Recall</td>
    <td>Context, Reference</td>
    <td>Reference Dependent</td>
    <td>Retriever’s ability to fetch the entities in the reference answer.</td>
  </tr>
  <tr>
    <td>Context Relevancy</td>
    <td>Context, Query</td>
    <td>Reference Free</td>
    <td>Retriever’s ability to retrieve context relevant to the user query.</td>
  </tr>
</table> 

## Evaluating RAG Generator

The generator, typically a large language model, takes the retrieved information and uses it to generate coherent and contextually appropriate responses, enhancing the quality and factual accuracy of the output. 

Evaluating the RAG generator is crucial because its performance determines the accuracy, relevance, and reliability of the final output. A poorly functioning generator can produce misleading or incoherent responses, even with perfectly retrieved context.

Some of the popular metrics for RAG Generator evaluation are faithfulness, hallucination, response relevancy, relevant noise sensitivity, irrelevant noise sensitivity.

<table border="1">
  <tr>
    <th>Metric</th>
    <th>Inputs</th>
    <th>Type</th>
    <th>Assesses</th>
  </tr>
  <tr>
    <td>Faithfulness</td>
    <td>Context, Response</td>
    <td>Reference Free</td>
    <td>How much is the generated response factually consistent with the retrieved context.</td>
  </tr>
  <tr>
    <td>Hallucination</td>
    <td>Context, Response</td>
    <td>Reference Free</td>
    <td>How much is the generated response factually inconsistent with the retrieved context.</td>
  </tr>
  <tr>
    <td>Response Relevancy</td>
    <td>Response, Query</td>
    <td>Reference Free</td>
    <td>How much relevant is the generated response to the user query.</td>
  </tr>
  <tr>
    <td>Relevant Noise Sensitivity</td>
    <td>Context, Response, Reference</td>
    <td>Reference Dependent</td>
    <td>Sensitivity to noise (incorrect claims) in relevant retrieved context.</td>
  </tr>
  <tr>
    <td>Irrelevant Noise Sensitivity</td>
    <td>Context, Response, Reference</td>
    <td>Reference Dependent</td>
    <td>Sensitivity to noise (incorrect claims) in irrelevant retrieved context.</td>
  </tr>
</table>
