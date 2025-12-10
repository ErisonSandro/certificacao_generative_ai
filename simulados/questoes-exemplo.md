# Questoes de Exemplo - Databricks Certified Generative AI Engineer Associate

Este arquivo contem questoes similares as do exame real para voce praticar.

---

## Questao 1 (do PDF oficial)

**Secao**: Data Preparation
**Objetivo**: Apply a chunking strategy for a given document structure and model constraints

Um Generative AI Engineer esta carregando 150 milhoes de embeddings em um banco de dados vetorial que suporta no maximo 100 milhoes.

**Quais DUAS acoes podem ser tomadas para reduzir a contagem de registros?**

A. Aumentar o tamanho do chunk do documento
B. Diminuir o overlap entre chunks
C. Diminuir o tamanho do chunk do documento
D. Aumentar o overlap entre chunks
E. Usar um modelo de embedding menor

**Resposta Correta**: A, B

**Explicacao**:
- A: Chunks maiores = menos chunks = menos embeddings
- B: Menos overlap = menos duplicacao = menos embeddings
- C: Incorreto - chunks menores geram MAIS embeddings
- D: Incorreto - mais overlap gera MAIS embeddings
- E: Incorreto - tamanho do modelo nao afeta numero de embeddings

---

## Questao 2 (do PDF oficial)

**Secao**: Data Preparation
**Objetivo**: Identify needed source documents that provide necessary knowledge and quality for a given RAG application

Um Generative AI Engineer esta avaliando as respostas de uma aplicacao GenAI voltada para clientes que auxilia na venda de pecas automotivas. A aplicacao requer que o cliente insira explicitamente `account_id` e `transaction_id` para responder perguntas. Apos o lancamento inicial, o feedback do cliente foi que a aplicacao respondia bem sobre detalhes de pedidos e faturamento, mas falhava em responder com precisao perguntas sobre envio e data prevista de chegada.

**Qual dos seguintes retrievers melhoraria a capacidade da aplicacao de responder essas perguntas?**

A. Criar um vector store que inclui as politicas de envio da empresa e termos de pagamento para todas as pecas automotivas
B. Criar uma tabela feature store com `transaction_id` como chave primaria que e populada com dados de fatura e data prevista de entrega
C. Fornecer dados de exemplo para datas previstas de chegada como dataset de tuning, entao periodicamente fazer fine-tuning do modelo para que tenha informacoes atualizadas de envio
D. Modificar o prompt de chat para incluir quando o pedido foi feito e instruir o modelo a adicionar 14 dias a isso, ja que nenhum metodo de envio deve exceder 14 dias

**Resposta Correta**: B

**Explicacao**:
- B: Feature store com dados especificos de transacao resolve o problema diretamente
- A: Politicas gerais nao fornecem datas especificas de entrega
- C: Fine-tuning nao resolve falta de dados reais de envio
- D: Hardcoding 14 dias nao e preciso e nao usa dados reais

---

## Questao 3 (do PDF oficial)

**Secao**: Data Preparation
**Objetivo**: Choose the appropriate Python package to extract document content from provided source data and format

Um Generative AI Engineer esta construindo uma aplicacao RAG que dependera de contexto recuperado de documentos fonte que foram escaneados e salvos como arquivos de imagem em formatos como .jpeg ou .png. Eles querem desenvolver uma solucao usando a menor quantidade de linhas de codigo.

**Qual pacote Python deve ser usado para extrair o texto dos documentos fonte?**

A. beautifulsoup
B. scrapy
C. pytesseract
D. pyquery

**Resposta Correta**: C

**Explicacao**:
- C: pytesseract e OCR para extrair texto de imagens (JPEG, PNG)
- A: beautifulsoup e para parsing HTML
- B: scrapy e para web scraping
- D: pyquery e para parsing HTML/XML

---

## Questao 4 (do PDF oficial)

**Secao**: Application Development
**Objetivo**: Select an embedding model context length based on source documents, expected queries, and optimization strategy

Um Generative AI Engineer esta criando uma aplicacao baseada em LLM. Os documentos para seu retriever foram divididos em chunks de no maximo 512 tokens cada. O Generative AI Engineer sabe que custo e latencia sao mais importantes que qualidade para esta aplicacao. Eles tem varios niveis de context length para escolher.

**Qual atendera sua necessidade?**

A. context length 512: menor modelo e 0.13GB com dimensao de embedding 384
B. context length 514: menor modelo e 0.44GB e dimensao de embedding 768
C. context length 2048: menor modelo e 11GB e dimensao de embedding 2560
D. context length 32768: menor modelo e 14GB e dimensao de embedding 4096

**Resposta Correta**: A

**Explicacao**:
- A: Context length minimo necessario (512), menor modelo (0.13GB), mais rapido e barato
- B, C, D: Modelos maiores e context length desnecessario aumentam custo e latencia

---

## Questao 5 (do PDF oficial)

**Secao**: Application Development
**Objetivo**: Select the best LLM based on the attributes of the application to be developed

Um Generative AI Engineer gostaria de construir uma aplicacao que pode atualizar um campo memo de cerca de um paragrafo para apenas uma frase resumida que mostra a intencao do memo, mas que cabe no frontend da aplicacao.

**Com qual categoria de tarefa de Processamento de Linguagem Natural eles devem avaliar potenciais LLMs para esta aplicacao?**

A. text2text Generation
B. Sentencizer
C. Text Classification
D. Summarization

**Resposta Correta**: D

**Explicacao**:
- D: Summarization e a tarefa de reduzir texto longo em versao mais curta
- A: text2text e muito generico
- B: Sentencizer divide texto em sentencas, nao resume
- C: Classification categoriza, nao resume

---

## Questoes Adicionais

### Questao 6

**Secao**: Design Applications
**Objetivo**: Select chain components for a desired model input and output

Voce precisa construir uma aplicacao que mantem contexto de conversas anteriores. Qual componente LangChain voce deve usar?

A. PromptTemplate
B. LLMChain
C. Memory
D. OutputParser

**Resposta Correta**: C

**Explicacao**: Memory components armazenam e recuperam historico de conversas.

---

### Questao 7

**Secao**: Data Preparation
**Objetivo**: Define operations to write chunked text into Delta Lake

Qual e a sequencia correta para escrever documentos chunked em Delta Lake?

A. Ler documentos → Chunking → Explode chunks → Escrever Delta Table
B. Chunking → Ler documentos → Escrever Delta Table → Explode chunks
C. Escrever Delta Table → Ler documentos → Chunking → Explode chunks
D. Ler documentos → Escrever Delta Table → Chunking → Explode chunks

**Resposta Correta**: A

**Explicacao**: Ordem logica: primeiro ler, depois processar (chunk), depois explodir array, depois escrever.

---

### Questao 8

**Secao**: Application Development
**Objetivo**: Implement LLM guardrails

Qual tecnica NAO e apropriada para implementar guardrails contra prompt injection?

A. Detectar padroes suspeitos como "ignore previous instructions"
B. Validar tamanho maximo de input
C. Aumentar temperatura do modelo
D. Usar prompt templates com separadores claros

**Resposta Correta**: C

**Explicacao**: Temperatura controla aleatoriedade, nao protege contra injection.

---

### Questao 9

**Secao**: Assembling and Deploying
**Objetivo**: Create and query a Vector Search index

Qual tipo de Vector Search index sincroniza automaticamente com uma Delta Table?

A. Direct Vector Access Index
B. Delta Sync Index
C. Manual Index
D. Streaming Index

**Resposta Correta**: B

**Explicacao**: Delta Sync Index sincroniza automaticamente com Delta Table fonte.

---

### Questao 10

**Secao**: Assembling and Deploying
**Objetivo**: Register model to Unity Catalog using MLflow

Qual URI voce usa para configurar Unity Catalog como registry no MLflow?

A. "databricks"
B. "unity-catalog"
C. "databricks-uc"
D. "uc"

**Resposta Correta**: C

**Explicacao**: `mlflow.set_registry_uri("databricks-uc")` configura Unity Catalog.

---

### Questao 11

**Secao**: Governance
**Objetivo**: Use masking techniques as guardrails

Qual biblioteca Python e especializada em detectar e anonimizar PII?

A. spaCy
B. Presidio
C. NLTK
D. TextBlob

**Resposta Correta**: B

**Explicacao**: Presidio da Microsoft e especializada em deteccao e anonimizacao de PII.

---

### Questao 12

**Secao**: Governance
**Objetivo**: Select guardrail techniques to protect against malicious user inputs

Quais DUAS tecnicas sao efetivas contra prompt injection?

A. Detectar padroes de instrucoes de sistema
B. Aumentar tamanho do contexto
C. Usar delimitadores fortes entre instrucoes e input do usuario
D. Reduzir temperatura do modelo
E. Aumentar max_tokens

**Resposta Correta**: A, C

**Explicacao**:
- A: Detectar padroes como "system:", "ignore instructions" ajuda
- C: Delimitadores separam claramente instrucoes de input malicioso
- B, D, E: Nao protegem contra injection

---

### Questao 13

**Secao**: Evaluation and Monitoring
**Objetivo**: Use inference logging to assess deployed RAG application

Onde o Databricks armazena logs de inference quando auto_capture_config esta habilitado?

A. MLflow Tracking
B. Unity Catalog tables
C. DBFS
D. S3 bucket

**Resposta Correta**: B

**Explicacao**: Inference logs sao salvos em tabelas do Unity Catalog especificadas no auto_capture_config.

---

### Questao 14

**Secao**: Evaluation and Monitoring
**Objetivo**: Select key metrics to monitor for specific LLM deployment

Quais TRES metricas sao mais importantes para monitorar latencia de um endpoint LLM?

A. P50 latency
B. Token count
C. P95 latency
D. Error rate
E. P99 latency

**Resposta Correta**: A, C, E

**Explicacao**: P50, P95 e P99 latency sao metricas padrao de latencia (mediana, 95 percentil, 99 percentil).

---

### Questao 15

**Secao**: Evaluation and Monitoring
**Objetivo**: Compare evaluation and monitoring phases

Qual e a PRINCIPAL diferenca entre evaluation e monitoring?

A. Evaluation usa metricas diferentes de monitoring
B. Evaluation e offline/pre-deploy, monitoring e online/post-deploy
C. Evaluation e mais caro que monitoring
D. Monitoring e opcional, evaluation e obrigatorio

**Resposta Correta**: B

**Explicacao**: Evaluation acontece antes do deploy com dataset fixo; Monitoring acontece em producao continuamente.

---

### Questao 16

**Secao**: Application Development
**Objetivo**: Write metaprompts that minimize hallucinations

Qual instrucao em um metaprompt e MAIS efetiva para reduzir alucinacoes?

A. "Seja criativo na sua resposta"
B. "Responda APENAS baseado no contexto fornecido"
C. "Use seu conhecimento geral"
D. "Expanda sua resposta com detalhes adicionais"

**Resposta Correta**: B

**Explicacao**: Limitar resposta ao contexto fornecido previne invencao de fatos.

---

### Questao 17

**Secao**: Assembling and Deploying
**Objetivo**: Identify how to serve LLM application that leverages Foundation Model APIs

Qual e a VANTAGEM de usar Foundation Model APIs vs. hospedar proprio modelo?

A. Maior controle sobre o modelo
B. Menor latencia
C. Nao precisa gerenciar infraestrutura de modelo
D. Modelos customizados

**Resposta Correta**: C

**Explicacao**: Foundation Model APIs sao gerenciados, voce nao precisa se preocupar com infraestrutura.

---

### Questao 18

**Secao**: Assembling and Deploying
**Objetivo**: Identify batch inference workloads and apply ai_query() appropriately

Quando voce deve usar ai_query() ao inves de Model Serving endpoint?

A. Aplicacoes real-time com latencia critica
B. Processamento em lote de grandes volumes de dados
C. Chatbots interativos
D. Aplicacoes que precisam de chains complexas

**Resposta Correta**: B

**Explicacao**: ai_query() e otimizado para batch processing, nao para real-time.

---

### Questao 19

**Secao**: Data Preparation
**Objetivo**: Use tools and metrics to evaluate retrieval performance

Qual metrica mede quao cedo o primeiro resultado relevante aparece nos resultados de busca?

A. Precision
B. Recall
C. Mean Reciprocal Rank (MRR)
D. F1 Score

**Resposta Correta**: C

**Explicacao**: MRR mede a posicao do primeiro resultado relevante (1/rank).

---

### Questao 20

**Secao**: Data Preparation
**Objetivo**: Explain the role of re-ranking in information retrieval

Qual e o papel do re-ranking no processo de retrieval?

A. Substituir o retrieval inicial
B. Melhorar a ordem dos resultados apos retrieval inicial
C. Reduzir o numero de documentos indexados
D. Acelerar o processo de busca

**Resposta Correta**: B

**Explicacao**: Re-ranking reordena resultados do retrieval inicial usando modelo mais preciso.

---

## Dicas para o Exame

1. **Leia cuidadosamente**: Algumas questoes tem multiplas respostas corretas
2. **Identifique palavras-chave**: "DUAS acoes", "PRINCIPAL diferenca", "MAIS efetiva"
3. **Elimine opcoes claramente erradas**: Reduz chances de erro
4. **Gerencie seu tempo**: ~2 minutos por questao
5. **Marque para revisao**: Use para questoes dificeis
6. **Pratique com PDF oficial**: As 5 questoes do PDF sao muito similares ao exame real

## Como Usar Este Material

1. Tente responder sem olhar as respostas
2. Verifique sua resposta
3. Leia a explicacao mesmo se acertou
4. Revisite areas onde errou
5. Refaca questoes periodicamente

---

[← Voltar ao README Principal](./README.md)
