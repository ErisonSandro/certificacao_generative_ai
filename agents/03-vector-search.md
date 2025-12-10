# Agente: Databricks Vector Search

## Objetivo
Dominar Databricks Vector Search para criar, gerenciar e consultar índices vetoriais para aplicações RAG.

## O que é Vector Search?

### Conceito
Vector Search permite buscar documentos similares baseado em similaridade semântica (não apenas palavras-chave).

### Como Funciona
```
1. Texto → Embedding Model → Vetor (array de números)
2. Armazenar vetores em Vector Index
3. Query → Embedding → Vetor de busca
4. Encontrar vetores mais similares (cosine similarity, etc.)
5. Retornar documentos correspondentes
```

### Por que Usar?
- **Busca semântica**: "Carro" encontra "automóvel", "veículo"
- **Multilíngue**: Funciona através de idiomas
- **Contexto**: Entende significado, não apenas palavras

## Componentes Principais

### 1. Vector Search Endpoint
Recurso computacional que hospeda índices.

```python
from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

# Criar endpoint (uma vez por workspace/projeto)
client.create_endpoint(
    name="my-vector-search-endpoint"
)
```

**Importante**: Um endpoint pode hospedar múltiplos índices.

### 2. Vector Search Index
Armazena embeddings e metadata para busca.

**Dois tipos**:
- **Delta Sync Index**: Sincroniza automaticamente com Delta Table
- **Direct Vector Access Index**: Você fornece vetores manualmente

## Delta Sync Index

### Quando Usar
- Documentos armazenados em Delta Lake
- Quer sincronização automática
- Databricks gerencia embeddings

### Criação

```python
# 1. Preparar Delta Table fonte
# Deve conter coluna com texto a ser vetorizado

# 2. Criar index
index = client.create_delta_sync_index(
    endpoint_name="my-vector-search-endpoint",
    index_name="catalog.schema.doc_embeddings_index",

    # Delta Table fonte
    source_table_name="catalog.schema.document_chunks",

    # Pipeline type
    pipeline_type="TRIGGERED",  # ou "CONTINUOUS"

    # Chave primária
    primary_key="chunk_id",

    # Coluna com texto para embeddings
    embedding_source_column="chunk_text",

    # Endpoint do embedding model
    embedding_model_endpoint_name="databricks-bge-large-en"
)
```

### Pipeline Types

#### TRIGGERED
- Sincroniza quando você pedir (manual)
- Use quando dados mudam pouco

```python
# Sincronizar manualmente
index.sync()
```

#### CONTINUOUS
- Sincroniza automaticamente
- Use quando dados mudam frequentemente
- Detecta mudanças no Delta Table

### Schema da Delta Table

```sql
CREATE TABLE catalog.schema.document_chunks (
    chunk_id STRING NOT NULL,        -- Primary key
    chunk_text STRING NOT NULL,      -- Texto para embedding
    doc_source STRING,               -- Metadata
    doc_type STRING,                 -- Metadata
    created_at TIMESTAMP            -- Metadata
)
USING DELTA
```

**Importante**: Pode incluir outras colunas (metadata) que serão indexadas junto.

## Direct Vector Access Index

### Quando Usar
- Já tem embeddings pré-computados
- Usa embedding model externo
- Precisa controle total

### Criação

```python
index = client.create_direct_access_index(
    endpoint_name="my-vector-search-endpoint",
    index_name="catalog.schema.precomputed_embeddings_index",

    # Chave primária
    primary_key="id",

    # Dimensão dos vetores
    embedding_dimension=1024,

    # Coluna com vetores
    embedding_vector_column="embedding",

    # Schema completo
    schema={
        "id": "string",
        "embedding": "array<float>",
        "text": "string",
        "metadata": "string"
    }
)
```

### Adicionar Vetores

```python
# Preparar dados
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    {"id": "1", "text": "Documento sobre produtos"},
    {"id": "2", "text": "Documento sobre preços"},
]

# Gerar embeddings
for doc in documents:
    doc['embedding'] = model.encode(doc['text']).tolist()

# Upsert no index
index.upsert(documents)
```

**Nota**: Direct Access requer você gerenciar embeddings.

## Consultas (Similarity Search)

### Busca por Texto
```python
# Vector Search automaticamente converte texto em vetor
results = index.similarity_search(
    query_text="Como configurar autenticação?",
    columns=["chunk_id", "chunk_text", "doc_source"],
    num_results=5
)

# Processar resultados
for result in results['data_array']:
    print(f"Score: {result['score']}")
    print(f"Text: {result['chunk_text']}")
    print(f"Source: {result['doc_source']}")
    print("---")
```

### Busca por Vetor
```python
# Se já tem embedding da query
query_embedding = model.encode("minha query").tolist()

results = index.similarity_search(
    query_vector=query_embedding,
    columns=["chunk_id", "chunk_text"],
    num_results=10
)
```

### Busca com Filtros
```python
# Filtrar por metadata
results = index.similarity_search(
    query_text="pricing",
    columns=["chunk_id", "chunk_text", "doc_type"],
    filters={"doc_type": "product_manual"},
    num_results=5
)
```

**Importante**: Filtros são aplicados ANTES da busca vetorial (mais eficiente).

## Similarity Metrics

### Cosine Similarity (padrão)
```
similarity = cos(θ) = (A · B) / (||A|| ||B||)
Range: -1 a 1 (geralmente 0 a 1)
1 = idênticos, 0 = não relacionados
```

### Euclidean Distance
```
distance = ||A - B||
Menor = mais similar
```

### Dot Product
```
similarity = A · B
Maior = mais similar
```

**No Databricks**: Cosine similarity é padrão e funciona bem para maioria dos casos.

## Integração com LangChain

```python
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings

# Embedding model
embeddings = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# Vector store
vector_store = DatabricksVectorSearch(
    endpoint=endpoint,
    index_name="catalog.schema.doc_index",
    embedding=embeddings,
    text_column="chunk_text"
)

# Usar como retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# Usar em chain
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

result = qa_chain({"query": "Qual o preço?"})
```

## Delta Sync vs Direct Access

| Aspecto | Delta Sync | Direct Access |
|---------|------------|---------------|
| **Embeddings** | Databricks gera | Você fornece |
| **Sincronização** | Automática | Manual |
| **Controle** | Menor | Total |
| **Complexidade** | Menor | Maior |
| **Use quando** | Dados no Delta Lake | Embeddings externos |

**Recomendação**: Use Delta Sync quando possível (mais simples).

## Atualização de Índices

### Delta Sync
```python
# Triggered
index.sync()  # Manual

# Continuous
# Automático, nada precisa fazer
```

### Direct Access
```python
# Upsert (insert ou update)
new_docs = [
    {"id": "3", "embedding": [...], "text": "novo documento"}
]
index.upsert(new_docs)

# Delete
index.delete(["id1", "id2"])
```

## Monitoramento e Manutenção

### Verificar Status do Index
```python
index_info = client.get_index(
    endpoint_name="my-endpoint",
    index_name="catalog.schema.my_index"
)

print(index_info.status)  # ONLINE, OFFLINE, etc.
print(index_info.index_type)  # DELTA_SYNC, DIRECT_ACCESS
```

### Verificar Endpoint
```python
endpoint = client.get_endpoint("my-endpoint")
print(endpoint.endpoint_status)
```

### Re-index (Delta Sync)
```python
# Força re-index completo
index.sync()
```

## Otimização de Performance

### 1. Escolha Embedding Model Apropriado
```python
# Modelos disponíveis no Databricks
models = [
    "databricks-bge-large-en",     # Inglês, alta qualidade
    "databricks-gte-large-en",     # Inglês, balanceado
    "databricks-bgte-m3",          # Multilíngue
]
```

**Trade-offs**:
- Modelos maiores: Melhor qualidade, mais lento, mais caro
- Modelos menores: Menor qualidade, mais rápido, mais barato

### 2. Otimize Número de Resultados
```python
# Não recupere mais que necessário
results = index.similarity_search(
    query_text=query,
    num_results=5  # Não 100!
)
```

### 3. Use Filtros
```python
# Filtros reduzem espaço de busca
results = index.similarity_search(
    query_text=query,
    filters={"category": "electronics"},  # Filtra ANTES de buscar
    num_results=5
)
```

### 4. Batch Queries
```python
# Se tem múltiplas queries, use batch
queries = ["query 1", "query 2", "query 3"]

# Mais eficiente que fazer uma por vez
```

## Casos de Uso Comuns

### Caso 1: RAG Simples
```python
# 1. Criar Delta Table com documentos
# 2. Criar Delta Sync Index
# 3. Usar em LangChain retriever
# 4. Conectar a LLM
```

### Caso 2: Busca com Metadados
```python
# Delta Table com metadata rica
"""
CREATE TABLE docs (
    id STRING,
    text STRING,
    category STRING,
    date DATE,
    author STRING,
    priority STRING
)
"""

# Buscar com filtros
results = index.similarity_search(
    query_text="urgent issues",
    filters={
        "priority": "high",
        "date": "> 2024-01-01"
    }
)
```

### Caso 3: Hybrid Search (Keyword + Semantic)
```python
# Combinar busca tradicional com vetorial
def hybrid_search(query, category=None):
    # 1. Vector search
    vector_results = index.similarity_search(
        query_text=query,
        filters={"category": category} if category else None,
        num_results=20
    )

    # 2. Keyword search (SQL)
    keyword_results = spark.sql(f"""
        SELECT * FROM docs
        WHERE text LIKE '%{query}%'
        LIMIT 20
    """)

    # 3. Merge e re-rank
    combined = merge_results(vector_results, keyword_results)

    return combined[:5]
```

## Troubleshooting

### Problema: Index não sincroniza
```python
# Verificar status
index_info = client.get_index(endpoint_name=..., index_name=...)
print(index_info.status)

# Forçar sync
index.sync()
```

### Problema: Baixa qualidade de resultados
- **Solução 1**: Melhorar chunking strategy
- **Solução 2**: Usar embedding model maior/melhor
- **Solução 3**: Adicionar re-ranking
- **Solução 4**: Aumentar num_results e filtrar depois

### Problema: Alta latência
- **Solução 1**: Reduzir num_results
- **Solução 2**: Usar filtros
- **Solução 3**: Cache results comuns
- **Solução 4**: Usar embedding model menor

## Custos

### Fatores de Custo
1. **Compute do Endpoint**: Tempo ativo
2. **Storage**: Tamanho dos embeddings
3. **Embedding Model**: Número de tokens processados

### Otimização de Custo
```python
# 1. Scale-to-zero quando possível
# 2. Use modelo menor adequado
# 3. Cache embeddings (Direct Access)
# 4. Filtre antes de vetorizar (se possível)
```

## Best Practices

### ✅ Faça
1. **Use Delta Sync** quando dados estão no Delta Lake
2. **Adicione metadata** útil para filtros
3. **Teste embedding models** diferentes
4. **Monitor qualidade** dos resultados
5. **Use filtros** para reduzir espaço de busca
6. **Versione seus índices** (catalog.schema.index_v1, v2...)

### ❌ Evite
1. **Índices muito grandes** sem filtros
2. **Embeddings muito grandes** (dimensão excessiva)
3. **Re-index desnecessários** (custoso)
4. **Ignorar metadata** (perde capacidade de filtro)
5. **Deixar endpoints ativos** quando não usa (custo!)

## Questões de Revisão

1. Qual a diferença entre Delta Sync e Direct Vector Access?
2. Quando usar TRIGGERED vs CONTINUOUS pipeline?
3. Como adicionar filtros em similarity search?
4. Que colunas são obrigatórias em Delta Sync index?
5. Como integrar Vector Search com LangChain?

## Exercícios Práticos

### Exercício 1: Delta Sync Index
1. Crie Delta Table com documentos
2. Crie Delta Sync Vector Search index
3. Sincronize o index
4. Faça queries de teste
5. Adicione novos documentos e re-sincronize

### Exercício 2: Filtros
Crie index com metadata (categoria, data, autor) e faça buscas com diferentes combinações de filtros.

### Exercício 3: Integração RAG
Integre Vector Search com LangChain para criar aplicação RAG completa.

## Comandos Essenciais

```python
# Criar endpoint
client.create_endpoint(name="endpoint")

# Criar Delta Sync index
client.create_delta_sync_index(...)

# Criar Direct Access index
client.create_direct_access_index(...)

# Sincronizar (Delta Sync)
index.sync()

# Buscar
index.similarity_search(query_text="...", num_results=5)

# Upsert (Direct Access)
index.upsert(documents)

# Deletar (Direct Access)
index.delete(ids)

# Verificar status
client.get_index(...)
```

## Recursos Adicionais

- [Databricks Vector Search Documentation](https://docs.databricks.com/generative-ai/vector-search.html)
- [Vector Search API Reference](https://docs.databricks.com/dev-tools/python-api.html)

---

[← Anterior: Chunking](./02-chunking-strategies.md) | [Próximo: Guardrails →](./04-guardrails.md)
