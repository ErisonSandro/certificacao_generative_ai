# 🚀 Preparação Intensiva - 4 Dias para a Prova

## ⚠️ Aviso Importante
Este é um plano **emergencial** para quem tem apenas 4 dias. O foco é nos tópicos de **maior peso** no exame.

## 📊 Distribuição do Exame (45 questões)

| Seção | Peso | Questões | Prioridade |
|-------|------|----------|------------|
| 1. Design Applications | 18% | ~8 questões | 🔥🔥🔥 ALTA |
| 2. Data Preparation | 22% | ~10 questões | 🔥🔥🔥 CRÍTICA |
| 3. Application Development | 27% | ~12 questões | 🔥🔥🔥 CRÍTICA |
| 4. Assembling & Deploying | 22% | ~10 questões | 🔥🔥🔥 CRÍTICA |
| 5. Governance | 7% | ~3 questões | 🔥 MÉDIA |
| 6. Evaluation & Monitoring | 4% | ~2 questões | 🔥 BAIXA |

## 🎯 Estratégia dos 4 Dias

### Regra de Ouro
**Foque nos tópicos específicos do Databricks** - são os que mais caem e diferenciam quem passa de quem não passa!

---

# 📅 DIA 1 - Fundamentos + Data Preparation (22% do exame)

**Tempo total**: 6-8 horas
**Objetivo**: Dominar chunking e prompt engineering

## Manhã (3-4 horas): Prompt Engineering

### Tópicos OBRIGATÓRIOS:
1. ✅ **Few-shot vs Zero-shot** (cai MUITO!)
   - Quando usar cada um
   - Exemplo de questão: "Modelo pequeno + formato custom" → Few-shot

2. ✅ **Chain-of-Thought (CoT)**
   - Cálculos e raciocínio
   - Adicionar "Pense passo a passo"

3. ✅ **Structured Output**
   - JSON schema em prompts
   - "Retorne APENAS JSON válido"

4. ✅ **Anti-Hallucination**
   - Limitar ao contexto
   - "Responda APENAS com informações do contexto"
   - Incluir referências [ref: doc_id]

5. ✅ **Delimitadores**
   - `"""texto"""` ou `<text>...</text>`
   - Proteção contra prompt injection

**Estude**: [Agente 01: Prompt Engineering](./agents/01-prompt-engineering.md)

**Exercício prático**:
```python
# Crie um prompt que:
# 1. Extrai nome, email, telefone de texto
# 2. Retorna JSON
# 3. Não alucina se info não existe
```

---

## Tarde (3-4 horas): Chunking Strategies

### Tópicos OBRIGATÓRIOS:

1. ✅ **Fixed-size Chunking**
   ```python
   chunk_size = 512  # tokens
   overlap = 50      # tokens
   ```
   - Quando usar: documentos uniformes
   - Trade-off: simplicidade vs qualidade

2. ✅ **Semantic Chunking**
   - Divide por significado (parágrafos, seções)
   - Melhor para retrieval, mais lento

3. ✅ **Recursive Chunking** (LangChain)
   ```python
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   splitter = RecursiveCharacterTextSplitter(
       chunk_size=500,
       chunk_overlap=50,
       separators=["\n\n", "\n", " ", ""]
   )
   ```

4. ✅ **Overlap** (CAI MUITO!)
   - Por que usar: contexto entre chunks
   - Típico: 10-20% do chunk_size
   - Overlap de 50 tokens em chunk de 512 = ~10%

5. ✅ **Otimização de Embeddings**
   - **Questão clássica**: "Você tem 150M embeddings, quer reduzir para 100M. Como?"
   - **Resposta**: Aumentar chunk_size de 512 para ~768 tokens
   - Cálculo: 150M → 100M = redução de 33% → chunk +33% maior

**Estude**: [Agente 02: Chunking Strategies](./agents/02-chunking-strategies.md)

**Exercício prático**:
- Se chunk_size=512 e overlap=64, quantos chunks em 10,000 tokens?
- Calcular: (10000-64) / (512-64) = ~22 chunks

---

## Noite (1 hora): Revisão

- [ ] Revisar diferenças: zero-shot vs few-shot
- [ ] Revisar: quando aumentar vs diminuir chunk_size
- [ ] Fazer 5 questões de exemplo da seção 1 e 2

---

# 📅 DIA 2 - Vector Search + Databricks (44% do exame!)

**Tempo total**: 6-8 horas
**Objetivo**: Dominar Vector Search e Model Serving (ESPECÍFICOS DATABRICKS!)

## Manhã (4 horas): Vector Search

### Tópicos OBRIGATÓRIOS:

1. ✅ **Delta Sync Index** (CAI MUITO!)
   ```python
   from databricks.vector_search.client import VectorSearchClient

   client = VectorSearchClient()

   index = client.create_delta_sync_index(
       endpoint_name="my-endpoint",
       index_name="catalog.schema.index",
       source_table_name="catalog.schema.docs",
       pipeline_type="TRIGGERED",  # ou CONTINUOUS
       primary_key="doc_id",
       embedding_source_column="text",
       embedding_model_endpoint_name="bge-large-en"
   )
   ```
   - Sincroniza automaticamente com Delta Table
   - Atualiza quando tabela muda

2. ✅ **Direct Vector Access Index**
   ```python
   index = client.create_direct_access_index(
       endpoint_name="my-endpoint",
       index_name="catalog.schema.direct_index",
       primary_key="id",
       embedding_dimension=768,  # depende do modelo
       embedding_vector_column="embedding"
   )
   ```
   - Você gerencia embeddings manualmente
   - Mais controle, mais trabalho

3. ✅ **Quando usar cada um?**
   | Cenário | Tipo |
   |---------|------|
   | Dados mudam frequentemente | Delta Sync |
   | Quer automação | Delta Sync |
   | Embedding customizado | Direct Access |
   | Controle total | Direct Access |

4. ✅ **Similarity Search**
   ```python
   results = index.similarity_search(
       query_text="Como funciona Vector Search?",
       columns=["id", "text", "metadata"],
       num_results=5
   )
   ```

5. ✅ **Similarity Search com Filtros**
   ```python
   results = index.similarity_search(
       query_text="pergunta",
       columns=["id", "text"],
       filters={"category": "electronics", "price": {"$lt": 1000}},
       num_results=10
   )
   ```
   - `$lt`: menor que
   - `$gt`: maior que
   - `$eq`: igual
   - Combinações: `{"$and": [...]}`

**Estude**: [Agente 03: Vector Search](./agents/03-vector-search.md)

**Questões típicas de exame**:
> "Você quer que embeddings sejam atualizados automaticamente quando documentos mudam. Qual index?"
> → **Delta Sync Index**

> "Você está usando um modelo de embedding customizado. Qual index?"
> → **Direct Vector Access Index**

---

## Tarde (3-4 horas): Model Serving + Unity Catalog

### Tópicos OBRIGATÓRIOS:

1. ✅ **Criar Endpoint**
   ```python
   from databricks.sdk import WorkspaceClient
   from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput

   w = WorkspaceClient()

   endpoint = w.serving_endpoints.create(
       name="rag-endpoint",
       config=EndpointCoreConfigInput(
           served_entities=[
               ServedEntityInput(
                   entity_name="catalog.schema.model",
                   entity_version="1",
                   workload_size="Small",  # Small, Medium, Large
                   scale_to_zero_enabled=True
               )
           ]
       )
   )
   ```

2. ✅ **Workload Size** (CAI NO EXAME!)
   | Size | Use quando |
   |------|------------|
   | Small | Tráfego baixo, dev/test |
   | Medium | Produção média |
   | Large | Alta demanda, baixa latência |

3. ✅ **Scale to Zero**
   - `True`: economiza custo, latência inicial maior
   - `False`: sempre ligado, sem cold start

4. ✅ **Registrar Modelo no Unity Catalog**
   ```python
   import mlflow

   mlflow.set_registry_uri("databricks-uc")

   with mlflow.start_run():
       mlflow.langchain.log_model(
           lc_model=chain,
           artifact_path="model",
           registered_model_name="catalog.schema.rag_model"
       )
   ```

5. ✅ **Aliases** (CAI!)
   ```python
   client = mlflow.tracking.MlflowClient()

   client.set_registered_model_alias(
       name="catalog.schema.model",
       alias="champion",  # ou "challenger"
       version="2"
   )

   # Carregar
   model = mlflow.pyfunc.load_model("models:/catalog.schema.model@champion")
   ```

6. ✅ **PyFunc** (IMPORTANTE!)
   ```python
   from mlflow.pyfunc import PythonModel

   class RAGModel(PythonModel):
       def load_context(self, context):
           # Carregar recursos (uma vez)
           self.retriever = load_retriever()
           self.llm = load_llm()

       def predict(self, context, model_input):
           # Chamado a cada request
           query = model_input['query'][0]
           docs = self.retriever.get_relevant_documents(query)
           response = self.llm.invoke(f"Context: {docs}\n\nQ: {query}")
           return {"answer": response}
   ```

7. ✅ **ai_query() para Batch Inference**
   ```sql
   CREATE OR REPLACE TABLE sentiments AS
   SELECT
     id,
     text,
     ai_query(
       'databricks-llama-2-70b-chat',
       CONCAT('Classify sentiment: ', text)
     ) AS sentiment
   FROM reviews;
   ```
   - **Quando usar**: batch processing, ETL
   - **Quando NÃO usar**: real-time, latência crítica

**Estude**: [Agente 05: Model Serving](./agents/05-model-serving-unity-catalog.md)

---

## Noite (1 hora): Revisão

- [ ] Delta Sync vs Direct Access - diferenças
- [ ] Workload sizes e quando usar
- [ ] PyFunc structure
- [ ] Fazer 10 questões seção 4

---

# 📅 DIA 3 - Application Development (27% - MAIOR SEÇÃO!)

**Tempo total**: 6-8 horas
**Objetivo**: LangChain, Guardrails, RAG

## Manhã (3 horas): LangChain Essentials

### Tópicos OBRIGATÓRIOS:

1. ✅ **RAG Chain**
   ```python
   from langchain.chains import RetrievalQA

   qa_chain = RetrievalQA.from_chain_type(
       llm=llm,
       retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
       return_source_documents=True
   )

   result = qa_chain({"query": "pergunta"})
   # result['result'] = resposta
   # result['source_documents'] = docs usados
   ```

2. ✅ **Sequential Chain**
   ```python
   from langchain.chains import SequentialChain, LLMChain

   # Chain 1: resumir
   chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="resumo")

   # Chain 2: analisar resumo
   chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="analise")

   seq_chain = SequentialChain(
       chains=[chain1, chain2],
       input_variables=["texto"],
       output_variables=["resumo", "analise"]
   )
   ```

3. ✅ **Agents com Tools**
   ```python
   from langchain.agents import create_react_agent, AgentExecutor
   from langchain.tools import Tool

   tools = [
       Tool(
           name="Search",
           func=search_function,
           description="Busca informações na base"
       ),
       Tool(
           name="Calculator",
           func=calculator,
           description="Faz cálculos"
       )
   ]

   agent = create_react_agent(llm, tools, prompt)
   agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

   result = agent_executor.invoke({"input": "Quanto é 25 * 4?"})
   ```

4. ✅ **Memory**
   ```python
   from langchain.memory import ConversationBufferMemory

   memory = ConversationBufferMemory()

   chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

   chain.run("Meu nome é João")
   chain.run("Qual é meu nome?")  # Lembra "João"
   ```

5. ✅ **Retrievers**
   ```python
   retriever = vector_store.as_retriever(
       search_type="similarity",  # ou "mmr"
       search_kwargs={
           "k": 5,
           "filter": {"category": "tech"}
       }
   )

   docs = retriever.get_relevant_documents("pergunta")
   ```

**Estude**: [Agente 07: LangChain](./agents/07-langchain-basics.md)

---

## Tarde (3 horas): Guardrails

### Tópicos OBRIGATÓRIOS:

1. ✅ **Input Guardrails - Prompt Injection**
   ```python
   import re

   def detect_prompt_injection(user_input):
       patterns = [
           r'ignore\s+(previous|all)\s+instructions',
           r'disregard.*instructions',
           r'you are now',
           r'forget everything'
       ]

       for pattern in patterns:
           if re.search(pattern, user_input, re.IGNORECASE):
               return True  # BLOQUEADO
       return False
   ```

2. ✅ **Output Guardrails - PII Detection**
   ```python
   import re

   def detect_pii(text):
       patterns = {
           'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
           'phone': r'\b\d{2,3}[-.\s]?\d{4,5}[-.\s]?\d{4}\b',
           'cpf': r'\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b'
       }

       found = {}
       for pii_type, pattern in patterns.items():
           matches = re.findall(pattern, text)
           if matches:
               found[pii_type] = matches

       return found
   ```

3. ✅ **PII Masking**
   ```python
   def mask_pii(text):
       # Email
       text = re.sub(
           r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
           '[EMAIL]',
           text
       )

       # Telefone
       text = re.sub(
           r'\b\d{2,3}[-.\s]?\d{4,5}[-.\s]?\d{4}\b',
           '[TELEFONE]',
           text
       )

       return text
   ```

4. ✅ **Toxicity Detection** (usando library)
   ```python
   from guardrails.hub import DetectToxicity

   guard = DetectToxicity(threshold=0.8, on_fail="exception")

   # Validar output
   validated = guard.validate(llm_output)
   ```

5. ✅ **Rate Limiting**
   ```python
   from functools import wraps
   import time

   class RateLimiter:
       def __init__(self, max_calls, period):
           self.max_calls = max_calls
           self.period = period
           self.calls = []

       def __call__(self, func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               now = time.time()
               self.calls = [c for c in self.calls if now - c < self.period]

               if len(self.calls) >= self.max_calls:
                   raise Exception("Rate limit exceeded")

               self.calls.append(now)
               return func(*args, **kwargs)
           return wrapper

   # Uso
   @RateLimiter(max_calls=10, period=60)  # 10 calls/min
   def call_llm(query):
       return llm.invoke(query)
   ```

**Estude**: [Agente 04: Guardrails](./agents/04-guardrails.md)

**Questões típicas**:
> "Como proteger contra usuários tentando fazer o LLM ignorar instruções?"
> → **Input guardrails com regex de prompt injection**

> "Seu LLM está vazando emails. Como corrigir?"
> → **Output guardrail com PII masking**

---

## Noite (2 horas): RAG End-to-End

**Pipeline completo**:
```python
# 1. Chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 2. Vector Search (Databricks)
from databricks.vector_search.client import VectorSearchClient
client = VectorSearchClient()
index = client.create_delta_sync_index(...)

# 3. Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 4. Prompt
from langchain import PromptTemplate
prompt = PromptTemplate(
    template="""
    Responda APENAS com informações do contexto.

    Contexto: {context}
    Pergunta: {question}

    Resposta:
    """,
    input_variables=["context", "question"]
)

# 5. Chain
from langchain.chains import RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 6. Guardrails
def safe_rag(query):
    # Input guardrail
    if detect_prompt_injection(query):
        return "Input bloqueado"

    # RAG
    result = qa_chain({"query": query})

    # Output guardrail
    answer = mask_pii(result['result'])

    return answer
```

---

# 📅 DIA 4 - Governance + Evaluation + REVISÃO GERAL

**Tempo total**: 6-8 horas
**Objetivo**: Completar tópicos menores e revisar tudo

## Manhã (2 horas): MLflow Evaluation

### Tópicos OBRIGATÓRIOS:

1. ✅ **mlflow.evaluate()**
   ```python
   import mlflow
   import pandas as pd

   eval_data = pd.DataFrame({
       'query': ['Qual o preço?', 'Como funciona?'],
       'ground_truth': ['R$ 100', 'Descrição...']
   })

   results = mlflow.evaluate(
       model=model_uri,
       data=eval_data,
       targets='ground_truth',
       model_type='text'
   )

   print(results.metrics)
   ```

2. ✅ **Métricas Principais**
   - **BLEU**: similaridade de n-grams (tradução, sumarização)
   - **ROUGE**: overlap de palavras (sumarização)
   - **Perplexity**: quão "surpreso" o modelo fica (menor = melhor)

3. ✅ **LLM-as-Judge**
   ```python
   def llm_judge(query, response):
       judge_prompt = f"""
       Avalie de 1-5:
       Q: {query}
       A: {response}

       Critérios: relevância, precisão, completude

       Score:
       """
       return judge_llm.invoke(judge_prompt)
   ```

4. ✅ **Inference Logging**
   ```python
   w.serving_endpoints.update_config(
       name="endpoint",
       auto_capture_config={
           "catalog_name": "catalog",
           "schema_name": "schema",
           "table_name_prefix": "endpoint"
       }
   )
   ```

**Estude**: [Agente 06: MLflow](./agents/06-mlflow-evaluation.md)

---

## Tarde (2 horas): Governance

### Tópicos OBRIGATÓRIOS:

1. ✅ **LGPD/GDPR Compliance**
   - PII masking (já viu em guardrails)
   - Right to be forgotten
   - Data minimization

2. ✅ **Unity Catalog Permissions**
   ```python
   w.grants.update(
       securable_type="FUNCTION",
       full_name="catalog.schema.model",
       changes=[
           {"principal": "data-scientists", "add": ["EXECUTE"]},
           {"principal": "analysts", "add": ["READ"]}
       ]
   )
   ```

3. ✅ **Audit Logging**
   - Quem acessou o modelo?
   - Quando?
   - Que dados?

---

## Tarde/Noite (4 horas): REVISÃO GERAL + QUESTÕES

### Checklist Final

#### Vector Search
- [ ] Delta Sync vs Direct Access - quando usar cada
- [ ] similarity_search() com filtros
- [ ] Integração com LangChain retriever

#### Model Serving
- [ ] Criar endpoint com workload size
- [ ] Scale to zero - trade-offs
- [ ] Registrar modelo no Unity Catalog
- [ ] Aliases (champion/challenger)
- [ ] PyFunc structure

#### Chunking
- [ ] Fixed-size vs semantic vs recursive
- [ ] Overlap - por que usar
- [ ] Otimizar embeddings (reduzir quantidade)

#### Prompts
- [ ] Zero-shot vs few-shot
- [ ] Chain-of-Thought
- [ ] Structured output (JSON)
- [ ] Anti-hallucination
- [ ] Delimitadores

#### LangChain
- [ ] RAG chain
- [ ] Sequential chain
- [ ] Agents com tools
- [ ] Memory

#### Guardrails
- [ ] Prompt injection detection
- [ ] PII masking
- [ ] Rate limiting

#### MLflow
- [ ] mlflow.evaluate()
- [ ] Inference logging
- [ ] Métricas (BLEU, ROUGE)

---

## 🎯 Faça TODAS as Questões de Exemplo

**Arquivo**: [questoes-exemplo.md](./questoes-exemplo.md)

Reserve **2 horas** para fazer as 20 questões e entender os erros.

---

## 📝 Folha de Cola - Leve Mentalmente

### Decisões Rápidas (Decorar!)

| Situação | Resposta |
|----------|----------|
| Modelo pequeno + formato custom | Few-shot prompting |
| Modelo grande + tarefa simples | Zero-shot |
| Cálculos e raciocínio | Chain-of-Thought |
| Output estruturado | JSON schema no prompt |
| Evitar alucinações | Limitar ao contexto |
| Proteger contra injection | Delimitadores |
| Dados mudam automaticamente | Delta Sync Index |
| Embedding customizado | Direct Access Index |
| Batch processing SQL | ai_query() |
| Real-time inference | Model Serving endpoint |
| Reduzir quantidade embeddings | Aumentar chunk_size |
| Melhorar retrieval quality | Diminuir chunk_size (mais granular) |
| Vazamento de PII | Output guardrail com masking |
| Usuário tenta hackear prompt | Input guardrail com regex |
| Dev/test com baixo tráfego | Small workload + scale_to_zero=True |
| Produção alta demanda | Large workload + scale_to_zero=False |

### Fórmulas

**Redução de embeddings**:
```
Se quer reduzir N% embeddings → aumentar chunk_size em N%
Exemplo: 150M → 100M = 33% redução → chunk +33%
```

**Overlap típico**:
```
10-20% do chunk_size
chunk_size=512 → overlap=50-100 tokens
```

**Número de chunks**:
```
chunks ≈ (total_tokens - overlap) / (chunk_size - overlap)
```

---

## ⚡ Dicas para o Dia da Prova

1. **Leia a pergunta 2 vezes** - muitas têm pegadinhas
2. **Identifique palavras-chave**:
   - "automático" → Delta Sync
   - "controle total" → Direct Access
   - "batch" → ai_query()
   - "real-time" → endpoint
   - "formato específico" → few-shot
   - "reduzir custo" → chunk_size maior, scale_to_zero

3. **Gerencie tempo**: 90min / 45 questões = 2 min/questão
   - Se não souber em 30s, marque e pule
   - Volte no final

4. **Chute inteligente**:
   - Elimine opções absurdas
   - Databricks sempre prefere soluções próprias (Vector Search, Unity Catalog, MLflow)
   - Em dúvida entre automatização vs manual → automatização

5. **Questões "select all that apply"**:
   - Geralmente 2-3 respostas corretas
   - Leia TODAS as opções

---

## 🚨 Tópicos que SEMPRE Caem

1. ✅ **Delta Sync vs Direct Access** (3-5 questões)
2. ✅ **Chunking strategies e overlap** (3-4 questões)
3. ✅ **Few-shot vs zero-shot** (2-3 questões)
4. ✅ **Unity Catalog model registry** (2-3 questões)
5. ✅ **PyFunc models** (2 questões)
6. ✅ **Guardrails (PII, injection)** (2-3 questões)
7. ✅ **ai_query() vs endpoints** (2 questões)
8. ✅ **Anti-hallucination prompts** (2 questões)

---

## 📚 Recursos Essenciais

- [Master Agent](./agents/00-master-agent.md) - Para dúvidas rápidas
- [Questões de Exemplo](./questoes-exemplo.md) - **FAÇA TODAS!**
- [Agente 02: Chunking](./agents/02-chunking-strategies.md) - Releia 2x
- [Agente 03: Vector Search](./agents/03-vector-search.md) - Releia 2x
- [Agente 05: Model Serving](./agents/05-model-serving-unity-catalog.md) - Releia 2x

---

## ✅ Checklist Pré-Prova (Noite Anterior)

- [ ] Revisei Delta Sync vs Direct Access
- [ ] Sei quando usar few-shot vs zero-shot
- [ ] Entendo PyFunc structure
- [ ] Sei calcular otimização de embeddings
- [ ] Conheço workload sizes
- [ ] Fiz todas 20 questões de exemplo
- [ ] Sei diferença ai_query() vs endpoint
- [ ] Entendo guardrails de input e output

---

## 💪 Você Consegue!

**Foco total nos 4 dias** = chances reais de passar!

Priorize:
1. Databricks-specific (Vector Search, Unity Catalog, Model Serving)
2. Chunking e prompts
3. Questões de exemplo

Boa sorte! 🚀

---

[← Voltar ao README Principal](./README.md)
