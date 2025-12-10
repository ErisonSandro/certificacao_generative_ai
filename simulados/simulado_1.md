# Questões do Simulado - Databricks Generative AI Engineer

## Pergunta 1:

No contexto do Databricks Generative AI, qual afirmação melhor descreve a relação entre IA, ML, DL e IA Generativa?

a) IA Generativa é um subconjunto de Deep Learning, que é um subconjunto de Machine Learning, que é um subconjunto de Inteligência Artificial.

b) Machine Learning é um superconjunto de Inteligência Artificial porque automatiza previsões, enquanto IA Generativa é independente de ambos.

c) Inteligência Artificial é um subconjunto de Machine Learning, e Deep Learning é um campo paralelo não relacionado à IA Generativa.

d) IA Generativa é um superconjunto de Inteligência Artificial, pois pode gerar saídas semelhantes às humanas além de previsão e classificação.

---

## Pergunta 2:

Você está projetando uma aplicação multi-modal alimentada por IA Generativa no Databricks que produz conteúdo personalizado em texto, visuais e áudio.

Qual das seguintes afirmações sobre aplicações de IA Generativa é MAIS precisa?

a) Modelos de IA Generativa para geração de texto como GPT-4 dependem principalmente de Redes Adversariais Generativas (GANs) para produzir respostas semelhantes às humanas.

b) GANs são amplamente usadas para geração de conteúdo visual porque envolvem uma arquitetura gerador-discriminador, enquanto modelos baseados em texto normalmente dependem de arquiteturas transformer.

c) IA Generativa para geração de áudio depende apenas de técnicas de aprendizado supervisionado treinadas exclusivamente em conjuntos de dados rotulados de música e fala.

d) Em pesquisa e desenvolvimento, IA generativa é principalmente limitada a relatórios automatizados; não pode ser usada para descoberta de medicamentos ou otimização de design de engenharia.

---

## Pergunta 3:

Você está construindo um chatbot de IA Generativa no Databricks que aproveita LLMs para inteligência conversacional.

Qual das seguintes afirmações sobre arquitetura LLM é MAIS precisa?

a) Transformers processam texto palavra por palavra em sequência estrita, tornando-os mais lentos, mas mais precisos para documentos longos em comparação com arquiteturas mais antigas baseadas em RNN.

b) O mecanismo de auto-atenção em transformers permite que os modelos foquem dinamicamente nas partes mais relevantes da entrada, tornando a compreensão de contexto significativamente melhor do que modelos de redes neurais mais antigos.

c) LLMs requerem redes neurais convolucionais (CNNs) como sua arquitetura principal para entender o contexto posicional em documentos grandes.

d) Ao contrário de arquiteturas mais antigas, transformers não podem lidar com processamento paralelo porque analisam um token por vez, tornando-os ineficientes para aplicações em tempo real.

---

## Pergunta 4:

Você está construindo um sistema de suporte ao cliente alimentado por LLM no Databricks. O modelo recebe a consulta do usuário:

"Por que meu pedido está atrasado?"

Qual das seguintes sequências melhor descreve como o LLM processa esta solicitação?

a) Prompt → Modelo → Tokenizador → Tarefa

b) Prompt → Tarefa → Tokenizador → Modelo

c) Prompt → Tokenizador → Modelo → Tarefa

d) Tokenizador → Prompt → Modelo → Tarefa

---

## Pergunta 5:

Você está construindo um sistema de suporte multilíngue alimentado por IA Generativa no Databricks. O sistema deve:

- Detectar o sentimento do cliente a partir do feedback,
- Resumir conversas,
- Traduzir respostas para diferentes idiomas, e
- Engajar em diálogo semelhante ao humano.

Qual das seguintes opções mapeia corretamente as tarefas de LLM para suas respectivas categorias?

a) Análise de sentimento → NLG, Resumo → NLU, Tradução → NLU, Diálogo → NLG

b) Análise de sentimento → NLU, Resumo → NLG, Tradução → NLG, Diálogo → NLG

c) Análise de sentimento → NLU, Resumo → NLU, Tradução → NLG, Diálogo → NLU

d) Análise de sentimento → NLG, Resumo → NLG, Tradução → NLU, Diálogo → NLU

---

## Pergunta 6:

Você está construindo uma aplicação LLM multilíngue no Databricks que suporta inglês, espanhol e chinês. Durante a implantação do modelo, os usuários relatam que palavras raras e compostas não estão sendo compreendidas corretamente.

Qual das seguintes afirmações sobre tokenização é MAIS precisa?

a) Tokenização converte tokens diretamente em embeddings sem atribuir IDs numéricos, o que permite que o modelo processe qualquer idioma sem estratégias adicionais.

b) Tokenizadores modernos como Byte Pair Encoding (BPE) e WordPiece dividem palavras raras ou desconhecidas em tokens de subpalavras, melhorando a capacidade do modelo de lidar com termos fora do vocabulário em vários idiomas.

c) Tokenização funciona apenas no nível de palavra; portanto, palavras desconhecidas ou raras não podem ser processadas por LLMs, a menos que sejam adicionadas manualmente ao vocabulário.

d) Tokenização é opcional em LLMs baseados em transformer, pois eles podem processar texto bruto legível por humanos diretamente.

---

## Pergunta 7:

Você está desenvolvendo uma ferramenta de geração de conteúdo de IA Generativa no Databricks. Durante os testes, você nota que o modelo produz respostas vagas e inconsistentes, mesmo que o LLM subjacente e o tokenizador estejam funcionando corretamente.

Qual das seguintes estratégias provavelmente melhoraria a qualidade e relevância das saídas do modelo?

a) Aumentar o número de parâmetros do modelo para melhorar a compreensão do contexto, independentemente da estrutura do prompt.

b) Refinar e otimizar o prompt com instruções detalhadas usando técnicas de engenharia de prompt, garantindo que o modelo tenha contexto suficiente para a tarefa desejada.

c) Mudar de modelos baseados em transformer para arquiteturas convolucionais, já que transformers não podem lidar com instruções ambíguas de forma eficaz.

d) Desabilitar a tokenização para deixar o modelo processar texto bruto legível por humanos, reduzindo a complexidade e melhorando a precisão da saída.

---

## Pergunta 8:

Um LLM do Databricks é implantado para classificar manchetes de notícias em categorias: Tecnologia, Finanças, Esportes e Política.

O modelo nunca viu exemplos rotulados para essas categorias, mas você fornece um prompt instruindo-o a escolher entre as categorias fornecidas.

a) Usa classificação zero-shot para atribuir rótulos com base no conhecimento geral.

b) Requer fine-tuning em um conjunto de dados rotulado para previsões precisas

c) Combina palavras-chave no texto com listas específicas de rótulos predefinidos

d) Aproveita embeddings treinados especificamente para cada domínio para classificar

---

## Pergunta 9:

Você está projetando um sistema de análise de sentimento personalizado usando um LLM do Databricks. Você fornece ao modelo três exemplos rotulados de tweets e pede para classificar um quarto:

Tweet1: Eu odeio filas longas → Sentimento: Negativo
Tweet2: Eu amo férias na praia → Sentimento: Positivo
Tweet3: O café está bom → Sentimento: Neutro

Tweet4: Este telefone superou minhas expectativas → Sentimento: ?

Como o LLM lida com esta situação?

a) Aprende com os poucos exemplos no prompt e prevê o sentimento com base no contexto.

b) Não pode classificar o sentimento a menos que seja retreinado em um conjunto de dados rotulado completo

c) Ignora os exemplos e realiza classificação zero-shot em vez disso

d) Tokeniza os tweets, mas não pode entender o contexto específico da tarefa sem fine-tuning

---

## Pergunta 10:

Você está usando um LLM hospedado no Databricks para resumir um relatório financeiro de 10 páginas para executivos. O resumo usa frases completamente novas, mas captura todos os principais insights com precisão.

Qual abordagem de resumo o LLM está provavelmente usando?

a) Resumo extrativo

b) Resumo abstrativo

c) Redução de sentenças tokenizadas

d) Resumo baseado em frequência

---

## Pergunta 11:

You are building a Databricks-powered customer feedback analyzer that categorizes product reviews into labels:

Positive, Negative, and Neutral.

You test two approaches:

Approach 1: You only specify the category labels in the prompt without providing any examples.

Approach 2: You include three labeled examples of reviews and their sentiments before asking the model to classify a new review.

Which statement best explains the difference between these two approaches?

a) Approach 1 uses zero-shot classification because the LLM predicts categories using its pre-trained knowledge without examples, while Approach 2 uses few-shot learning since it leverages a handful of examples to guide classification.

b) Approach 1 and Approach 2 both represent few-shot learning, as providing labels in the prompt is equivalent to giving examples.

c) Approach 1 uses zero-shot classification because labels are predefined, but Approach 2 still relies on zero-shot classification since the examples are not part of model training.

d) Approach 1 uses few-shot learning since labels act as implicit examples, while Approach 2 uses fine-tuning because providing examples changes model behavior.

---

## Pergunta 12:

You are designing a customer support assistant in Databricks that uses a Retrieval-Augmented Generation (RAG) workflow.

A user asks:

"What is the refund policy for premium customers?"

The answer is available only in your company's internal policy documents, which the base LLM has not seen during training.

Which sequence best describes how the RAG workflow processes this query?

a) The LLM generates a response directly based on its training data without retrieving any additional information.

b) The system retrieves relevant policy documents from your vector database, combines them with the user's query to create an enhanced prompt, and passes it to the LLM, which then generates a grounded response.

c) The documents are fine-tuned into the base model, and the model directly answers based on its updated training.

d) The LLM retrieves relevant information from its internal training dataset rather than searching your own company documents.

---

## Pergunta 13:

You are building multiple AI-powered applications using Databricks and need to decide where Retrieval-Augmented Generation (RAG) provides the most value.

Which scenario demonstrates the best use case for RAG?

a) Generating fully fictional creative stories where no factual grounding is required.

b) Building a customer support chatbot that needs to fetch the latest product policies and provide accurate, real-time answers.

c) Training a new LLM from scratch using your company's historical datasets.

d) Using the LLM's internal training data to answer questions without referencing any external documents.

---

## Pergunta 14:

Which of the following best describes the sequence and role of components in a Retrieval-Augmented Generation (RAG) workflow?

a) Embeddings convert text into vectors, vector databases store these vectors, search and retrieval locates the most relevant ones, and prompt augmentation merges them with the user's query.

b) Vector databases generate embeddings, prompt augmentation creates queries, search and retrieval stores vectors, and embeddings merge results with the prompt.

c) Embeddings answer user queries directly, vector databases clean raw data, search and retrieval builds embeddings, and prompt augmentation stores them for reuse.

d) Embeddings act as the LLM itself, vector databases generate responses, search and retrieval checks accuracy, and prompt augmentation summarizes the answer.

---

## Pergunta 15:

In a RAG workflow, why are both document embeddings and query embeddings necessary?

a) Document embeddings create a searchable knowledge base during preparation, while query embeddings convert each user query into the same vector format for real-time semantic comparison.

b) Document embeddings generate the final answers, and query embeddings are used only for summarization before returning results.

c) Document embeddings translate queries into vectors, and query embeddings clean the documents before storage.

d) Document embeddings are updated every time a query is asked, while query embeddings are generated only once during system setup.

---

## Pergunta 16:

Why is a vector store essential in a Retrieval-Augmented Generation (RAG) workflow?

a) It stores embeddings and enables fast semantic search by finding vectors with similar meaning, rather than relying only on exact keyword matches.

b) It generates embeddings from documents and user queries before sending them to the LLM.

c) It replaces the need for embeddings, since text can be directly stored and searched with exact string matching.

d) It is used only as a caching layer for temporary query results in a RAG pipeline.

---

## Pergunta 17:

In a Retrieval-Augmented Generation (RAG) workflow, what is the main purpose of prompt augmentation?

a) To merge the user's original query with retrieved context, providing the LLM with in-context learning that enables accurate, source-backed answers.

b) To retrain the LLM with new documents so it permanently learns company-specific knowledge.

c) To embed the user's query into vectors so it can be compared against the document embeddings.

d) To replace the vector store by directly storing documents inside the LLM's prompt history.

---

## Pergunta 18:

Which of the following BEST describes the key role of correct chunking in preparing data for a Retrieval Augmented Generation (RAG) system?

a) It ensures that the entire document is embedded at once, maximizing context with a single vector.

b) It balances context retention and retrieval precision by breaking documents into appropriately sized, semantically coherent pieces.

c) It ensures faster document loading by using purely fixed-size chunks without overlap.

d) It removes the need for embedding models by relying solely on pre-defined document indices.

---

## Pergunta 19:

You are preparing data for a Retrieval Augmented Generation (RAG) system on Databricks. Your users frequently ask short, specific questions and you want embeddings that capture precise meanings within the content.

Which chunking strategy would be most effective in this scenario?

a) Sentence-based chunks

b) Paragraph-based chunks

c) Overlapping chunks

d) Windowed summarization

---

## Pergunta 20:

You are preparing a Retrieval Augmented Generation (RAG) pipeline in Databricks. After chunking the documents, you need to choose an embedding model.

Which of the following factors is most critical when selecting the right embedding model for your application?

a) The number of users who will query the system

b) Matching the model to your data properties, domain, and content type

c) Always choosing the largest available embedding model for better accuracy

d) Selecting the model solely based on cost to minimize infrastructure expenses

---

## Pergunta 21:

In a RAG workflow, why is vector search essential after converting documents and queries into embeddings?

a) Vector search only ranks documents based on their creation date, ignoring semantic meaning.

b) Vector search generates embeddings from documents and queries before sending them to the LLM for processing.

c) Vector search replaces the need for embeddings since it directly compares raw text strings for faster retrieval.

d) Vector search finds semantically relevant documents by comparing the similarity between vector embeddings, enabling results that understand meaning rather than just exact keyword matches.

---

## Pergunta 22:

In a RAG pipeline, which option correctly describes how vector search works end-to-end and why it outperforms keyword search?

a) Convert the user query to an embedding using the same model as the document chunks, compute similarity scores against stored document embeddings, rank and retrieve the top-k chunks, then assemble those chunks with the original question for the LLM. This works better than keyword search because embeddings capture semantic meaning (e.g., "canine care" ≈ "dog training").

b) Tokenize the query, run a BM25 keyword match, sort by recency, and send the most recent documents to the LLM without additional context assembly.

c) Fine-tune the LLM on the document set; at inference time, the model uses its training weights to recall relevant knowledge—no separate retrieval step is needed.

d) Store raw text in a relational database and use SQL LIKE filters to find matches; send matching rows directly to the LLM.

---

## Pergunta 23:

In a RAG workflow, what is the primary role of a vector database, and why is it preferred over a traditional database for storing embeddings?

a) A vector database replaces the need for embeddings by converting all documents to keywords and filtering them using metadata only.

b) A vector database stores raw text documents and retrieves them using traditional string matching techniques, improving retrieval speed without needing embeddings.

c) A vector database stores embeddings and metadata, uses specialized indices to retrieve vectors based on mathematical similarity rather than exact matches, and integrates with RAG pipelines for semantic search.

d) A vector database primarily fine-tunes the LLM on stored documents so that retrieval is no longer required during inference.

---

## Pergunta 24:

In a RAG workflow, when would you prefer using a vector library like FAISS instead of a vector database?

a) When your dataset is small enough to fit in memory, doesn't change frequently, and you need fast similarity searches without deploying a full database system.

b) When your dataset is large, highly dynamic, and requires real-time updates, filtering, and multi-user access.

c) When you need to fine-tune the LLM on your entire dataset before performing retrieval, bypassing similarity search altogether.

d) When your RAG application only relies on exact keyword matches rather than semantic similarity between embeddings.

---

## Pergunta 25:

In a RAG pipeline, why would you add a reranking step after performing a vector similarity search?

a) To evaluate retrieved document chunks beyond their similarity scores and rank them based on actual relevance to the user's query before sending them to the LLM.

b) To generate embeddings for documents and queries, ensuring the vectors are properly aligned before similarity search.

c) To fine-tune the LLM on the retrieved documents, so it learns their content permanently and doesn't need retrieval in future queries.

d) To replace vector search entirely, because reranking directly retrieves documents without using embeddings.

---

## Pergunta 26:

A Generative AI Engineer is building a question-answering system in Databricks. They notice that the LLM often hallucinates when answering fact-based queries that require referencing internal company policies. To improve factual recall, what should they do?

a) Increase the temperature setting of the LLM so that it generates more creative responses.

b) Fine-tune the LLM immediately on all company policies.

c) Pass relevant context documents into the LLM prompt during query processing.

d) Limit the LLM's maximum token output so that responses stay shorter and less likely to be wrong.

---

## Pergunta 27:

When working with Large Language Models (LLMs), passing in a very long context window can introduce certain downsides. Which of the following correctly describes these challenges?

a) Higher API Cost – More tokens in the input mean higher cost for each request.

b) Longer Inference Time – Processing more tokens increases completion latency.

c) Lost-in-the-Middle Problem – Content placed in the middle of the prompt may be ignored, leading to missed facts.

d) All of the other options are correct

---

## Pergunta 28:

A Generative AI Engineer is building a Retrieval-Augmented Generation (RAG) system. They embed all documents with one model and the user queries with another embedding model. The retrieved results are consistently irrelevant. What is the most likely cause?

a) Different Embedding Spaces – Queries and documents were embedded using different models, leading to mismatched vector spaces.

b) Chunk Size Too Large – The documents were chunked into large sections, reducing retrieval accuracy.

c) Incorrect Similarity Metric – The engineer used cosine similarity instead of Euclidean distance.

d) Insufficient Tokens in Query – The user query is too short for the model to generate accurate embeddings.

---

## Pergunta 29:

A Generative AI Engineer is tasked with evaluating the performance of a Retrieval-Augmented Generation (RAG) system. They know evaluation must be done at both the component level and the system level. Which of the following evaluation breakdowns is most appropriate?

a) Evaluate only the generator for fluency and coherence, since it produces the final output.

b) Evaluate chunking, embedding model, vector store, retrieval/re-ranker, and generator, both individually and as part of the full pipeline.

c) Evaluate only the retriever and generator, since chunking and embedding models are preprocessing steps.

d) Evaluate only the vector store performance, as it determines query latency and accuracy.

---

## Pergunta 30:

You are running weekend batch jobs that make multiple LLM calls through Databricks. During peak usage, the system hits API throttling limits, causing failures. You want a robust approach to avoid throttling issues in future runs while still tracking and deploying your models efficiently. Which is the best solution?

a) Use MLflow pyfunc to package the LLM workflow into a model and manage calls with built-in retry/backoff mechanisms.

b) Increase the LLM temperature parameter in prompt engineering so that requests are processed faster.

c) Schedule weekend runs at random times to avoid throttling without modifying the pipeline.

d) Disable MLflow tracking and log calls manually to reduce overhead.

---

## Pergunta 31:

You are deploying a Retrieval-Augmented Generation (RAG) application into production on Databricks. The app must call external LLM APIs securely without risking leaked credentials. Which is the best authentication practice to implement?

a) Store a static API token directly in the application code for simplicity.

b) Use an OAuth 2.0 flow with token issuance and automatic rotation.

c) Share the service principal credentials with all developers and rotate them manually once a year.

d) Embed the API key in a configuration file on the production VM without encryption.

---

## Pergunta 32:

You are building a RAG (Retrieval-Augmented Generation) application that queries both a Databricks SQL Warehouse and a Databricks Vector Search index. To ensure that end-user identities are preserved and data access policies are enforced consistently, the team wants to implement authentication passthrough. What does this mean, and which approach is the most appropriate?

a) Authentication passthrough means embedding a shared API token in the application code so that all queries run under a single service identity.

b) Authentication passthrough means the application forwards the end-user's identity to downstream systems so that access control policies are applied consistently. This can be achieved using mechanisms like SSO tokens or delegated credentials.

c) Authentication passthrough means generating a static database password for each user and storing it in the Databricks secret store.

d) Authentication passthrough means bypassing authentication checks for faster query execution since the application already authenticated once.

---

## Pergunta 33:

You are implementing a RAG pipeline and want to use a PromptTemplate to format the input before sending it to an LLM in the databricks. The following code snippet throws an error:

```python
from langchain.prompts import PromptTemplate

prompt_template = "Summarize the following document:\n{document}"

template = PromptTemplate(
    input_variables=["document"],
    template=prompt_template
)

# Error occurs here
formatted_prompt = template.format()
```

What is the correct way to call the template to avoid the error?

a) formatted_prompt = template.format(document="This is my text")

b) formatted_prompt = template.format(document="This is my text", model="gpt-4")

c) formatted_prompt = template.format()

d) formatted_prompt = template.format({"model": "gpt-4"})

---

## Pergunta 34:

You are deploying an LLM application with guardrails configured at the endpoint level to block responses containing PII (Personally Identifiable Information). A user sends a query containing their phone number. What happens to the request and what gets recorded in the inference table?

a) The request is fully processed, the phone number is stored in the inference table, and a normal response is returned.

b) Nothing is stored in the inference table because the guardrail blocks the request before logging.

c) No response is returned to the user, but a log entry indicating a blocked request due to PII is stored in the inference table.

d) The phone number is masked and both the masked value and the response are stored in the inference table.

---

## Pergunta 35:

You are designing an LLM-powered application that requires prompt templates, chaining of multiple model calls, and integration with a vector database for retrieval. Which tool is best suited for building this LLM workflow?

a) LangChain

b) Apache Spark

c) TensorFlow

d) Delta Lake

---

## Pergunta 36:

You are implementing a RAG pipeline with vector search. To improve the relevance of retrieved results, you add a reranking model that reorders the top-k candidates before passing them to the LLM. Which of the following is NOT a real benefit of reranking?

a) Improves the overall quality and relevance of the retrieved documents by scoring them more precisely.

b) Reduces the chance of hallucinations by providing the LLM with more contextually relevant inputs.

c) Helps optimize latency, since the reranking step makes the query return faster.

d) Provides a layer of personalization, as rerankers can be tuned with user-specific signals.

---

## Pergunta 37:

You are comparing different approaches for approximate nearest-neighbor (ANN) search in a vector database, such as LSH (Locality Sensitive Hashing) and HNSW (Hierarchical Navigable Small World graphs). The goal is to achieve high semantic accuracy when retrieving embeddings. Which similarity measure is most appropriate in this context?

a) Cosine similarity

b) ROUGE score

c) BLEU score

d) Levenshtein distance

---

## Pergunta 38:

You are building a news-based RAG application. The requirement is that if a user asks:

"Tell me what happened to XYZ around April 1, 2002?"

the system should only return documents from ±5 days of the query date (March 27, 2002 to April 5, 2002).

What is the best way to implement this requirement?

a) Retrieve all documents about XYZ regardless of date and rely on the LLM prompt to filter results.

b) Apply a metadata filter at retrieval so only documents with event_date BETWEEN '2002-03-27' AND '2002-04-05' are included.

c) Store pre-summarized versions of all documents so the date filter is not necessary.

d) Use cosine similarity in the vector database to automatically discard documents older than 5 days.

---

## Pergunta 39:

You are preparing documents for a RAG pipeline. The data sources include raw HTML pages as well as scanned PDFs that require OCR. You want to extract clean text from these unstructured formats before chunking and embedding. Which library is the most appropriate for this task?

a) BeautifulSoup

b) Unstructured

c) Tensor Flow

d) NLTK

---

## Pergunta 40:

You are designing a prompt for an LLM that processes user emails such as:

"Hi, can you update me for the order, orderid = 1234?"

The response must always be in valid JSON with the following fields:
- customer_id
- order_id
- order_date
- order_status

Which is the best prompt to ensure the response is both structured and reliable?

a) Extract the order details mentioned in the email and provide an answer in JSON format.

b) Return only valid JSON with keys: customer_id, order_id, order_date, order_status. Do not include any extra text or explanation outside the JSON.

c) Give me the order update in JSON format including the requested fields, and also provide a short English explanation after the JSON.

d) Summarize the order details in JSON, but make sure it looks human-readable and conversational.

---

## Pergunta 41:

You have a Delta table called etl_error_logs with a column error_message. The messages contain variations like:

"File not found: /mnt/data/a.csv"
"Missing file: /mnt/data/b.csv"
"FileNotFoundError on /mnt/data/c.csv"

You want to use the Databricks built-in AI functions in SQL to summarize these messages so that all these message can fall under "Missing File" category.

What could be your approach?

a) Use a prompt and mention - Summarize the error messages.

b) Group the error messages into categories and return the results as plain text.

c) For each row use regex to remove specific details e.g. file name, IDs etc and sanitize the error message. Then using ai_summary() function to summarize the sanitized text.

d) Rewrite the error messages in simpler English sentences.

---

## Pergunta 42:

A Generative AI Engineer has built a Retrieval-Augmented Generation (RAG) application that answers reader questions about a sci-fi book series on the publisher's online community forum.

The sci-fi novel manuscripts have been chunked, embedded, and stored in a vector database along with metadata like chapter name, section ID, and book title. These chunks are retrieved based on the user's query and sent to an LLM for generating responses.

Initially, the engineer chose the chunking strategy and configuration parameters based on intuition, but now they want a data-driven approach to select the optimal chunk size and overlap settings.

Which TWO strategies would help the engineer methodically optimize their chunking strategy and related parameters? (Choose two.)

a) Switch to a different embedding model and compare how the retrieval performance changes.

b) Build a query classifier that predicts which specific book is most likely to contain the answer and filter retrieval accordingly.

c) Select a quantitative evaluation metric (e.g., recall or NDCG) and experiment with different chunking strategies — such as splitting chunks by paragraphs, sections, or chapters — then choose the approach that achieves the best performance metric.

d) Use a set of known questions and correct answers, pass them to an LLM, and instruct it to suggest the optimal token counts for chunk size. Then, use a summary statistic (e.g., mean or median) to finalize the chunking configuration.

e) Create an LLM-as-a-judge evaluation metric that scores how well retrieved chunks answer previous questions and adjust chunking settings based on its feedback.

---

## Pergunta 43:

You are processing a large PDF document for a RAG pipeline. The text is extracted, split into smaller chunks, and stored as an array of chunks in a PySpark DataFrame. What is the best way to implement chunking for scalability and downstream use in vector search?

a) Use Auto Loader with a custom UDF to split the PDF text into chunks and then explode the array to flatten into individual rows.

b) Store the entire PDF as a single string column in a Delta table without chunking, and let the LLM handle splitting at query time.

c) Use a Python for loop outside Spark to manually split each document and then parallelize the result back into Spark.

d) Save the raw PDF binary into a Delta table and rely on vector databases to chunk automatically at ingestion time.

---

## Pergunta 44:

A Generative AI Engineer is building a Retrieval-Augmented Generation (RAG) application to help users learn aviation safety guidelines by answering questions about technical regulations.

Which sequence of steps correctly represents the workflow for building, evaluating, and deploying this RAG application?

a) Ingest documents from a source → Index the documents and save them to Vector Search → User submits a query → LLM retrieves relevant documents → Evaluate the model → LLM generates a response → Deploy using Model Serving

b) Ingest documents from a source → Index the documents and save them to Vector Search → User submits a query → LLM retrieves relevant documents → LLM generates a response → Evaluate the model → Deploy using Model Serving

c) Ingest documents from a source → Index the documents and save them to Vector Search → Evaluate the model → Deploy using Model Serving

d) User submits a query → Ingest documents from a source → Index the documents and save them to Vector Search → LLM retrieves relevant documents → LLM generates a response → Evaluate the model → Deploy using Model Serving

---

## Pergunta 45:

A Generative AI Engineer is developing a document retrieval system for a legal firm. Each document is a long legal contract, often exceeding 10,000 words.

The model used in the RAG pipeline has a context window limit of 512 tokens per chunk.

To ensure optimal retrieval accuracy and context preservation, which chunking strategy should the engineer implement?

a) Chunk the document into large sections of 1,000 words each.

b) Divide the document into chunks of 512 tokens each with no overlap.

c) Chunk the document into 512-token chunks with an overlap of 50 tokens.

d) Create chunks of 256 tokens each with an overlap of 100 tokens.
