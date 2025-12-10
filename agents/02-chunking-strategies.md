# Agente: Chunking Strategies

## Objetivo
Dominar estratégias de chunking (divisão de documentos) para otimizar aplicações RAG, incluindo seleção de tamanho, overlap e técnicas baseadas em estrutura de documentos.

## Por que Chunking é Importante?

### Problema
- LLMs têm limite de tokens (context window)
- Documentos são geralmente muito grandes
- Embedding models têm limites de tamanho
- Vector databases têm limites de capacidade

### Solução
Dividir documentos em pedaços (chunks) menores e gerenciáveis.

### Impacto
- **Muito pequeno**: Perde contexto, fragmenta informação
- **Muito grande**: Excede limites, recuperação imprecisa
- **Ideal**: Balanceia contexto e precisão

## Fatores que Influenciam Chunking

### 1. Context Window do Embedding Model
```python
# Exemplo: Modelo com context length 512 tokens
# Chunks devem ser <= 512 tokens

max_chunk_size = 512  # tokens
```

### 2. Tipo de Documento
- **Narrativo**: Parágrafos, capítulos
- **Técnico**: Seções, sub-seções
- **Estruturado**: Tabelas, listas
- **Código**: Funções, classes

### 3. Tipo de Queries Esperadas
- **Específicas**: Chunks menores, mais precisos
- **Gerais**: Chunks maiores, mais contexto

### 4. Limitações do Vector Database
```python
# Exemplo do exame:
# 150M embeddings mas database suporta 100M
# Solução: Aumentar chunk size ou diminuir overlap
```

## Estratégias Principais

### 1. Fixed-Size Chunking
Divide texto em pedaços de tamanho fixo.

```python
def fixed_size_chunking(text, chunk_size=512, overlap=50):
    """
    Divide texto em chunks de tamanho fixo com overlap

    Args:
        text: Texto a ser dividido
        chunk_size: Tamanho de cada chunk em caracteres
        overlap: Sobreposição entre chunks

    Returns:
        Lista de chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Move com overlap

    return chunks

# Exemplo de uso
text = "Texto muito longo..." * 1000
chunks = fixed_size_chunking(text, chunk_size=512, overlap=50)
print(f"Total de chunks: {len(chunks)}")
```

**Vantagens**:
- Simples de implementar
- Previsível em tamanho
- Rápido

**Desvantagens**:
- Pode quebrar no meio de frases
- Ignora estrutura do documento
- Pode separar informações relacionadas

**Quando usar**: Textos uniformes, necessidade de velocidade

### 2. Semantic Chunking
Divide por unidades semânticas (parágrafos, seções).

```python
def semantic_chunking(text, min_chunk_size=100):
    """
    Divide texto por parágrafos
    """
    # Dividir por parágrafos
    chunks = text.split('\n\n')

    # Filtrar chunks muito pequenos
    chunks = [c.strip() for c in chunks if len(c.strip()) > min_chunk_size]

    # Combinar chunks muito pequenos
    combined_chunks = []
    current_chunk = ""

    for chunk in chunks:
        if len(current_chunk) + len(chunk) < 1000:
            current_chunk += "\n\n" + chunk if current_chunk else chunk
        else:
            if current_chunk:
                combined_chunks.append(current_chunk)
            current_chunk = chunk

    if current_chunk:
        combined_chunks.append(current_chunk)

    return combined_chunks

# Uso
text = """
Parágrafo 1 sobre produto.

Parágrafo 2 sobre preço.

Parágrafo 3 sobre garantia.
"""
chunks = semantic_chunking(text)
```

**Vantagens**:
- Respeita estrutura natural
- Mantém contexto semântico
- Chunks mais coerentes

**Desvantagens**:
- Tamanhos variáveis
- Pode gerar chunks muito grandes ou pequenos
- Mais lento que fixed-size

**Quando usar**: Documentos bem estruturados, qualidade > velocidade

### 3. Recursive Character Text Splitting
LangChain's approach - tenta manter semântica usando hierarquia de separadores.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=[
        "\n\n",  # Tenta dividir por parágrafos primeiro
        "\n",    # Depois por linhas
        ". ",    # Depois por sentenças
        " ",     # Depois por palavras
        ""       # Por último, por caracteres
    ],
    length_function=len
)

chunks = text_splitter.split_text(text)
```

**Como funciona**:
1. Tenta dividir pelo primeiro separador (\n\n)
2. Se chunk ainda muito grande, tenta próximo separador
3. Continua recursivamente até chunk size adequado

**Vantagens**:
- Balanceia tamanho e semântica
- Configurável
- Mantém contexto quando possível

**Desvantagens**:
- Mais complexo
- Pode ser lento

**Quando usar**: Melhor opção geral para maioria dos casos

### 4. Document-Specific Chunking
Chunking baseado na estrutura específica do documento.

#### Para Markdown/HTML
```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

chunks = markdown_splitter.split_text(markdown_text)
```

#### Para Código
```python
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

# Python code splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=512,
    chunk_overlap=50
)

chunks = python_splitter.split_text(python_code)
```

#### Para PDF com Seções
```python
def chunk_by_sections(pdf_pages):
    """
    Divide PDF por seções detectadas
    """
    chunks = []
    current_section = []

    for page in pdf_pages:
        text = page.extract_text()

        # Detectar início de nova seção (heurística)
        if is_section_start(text):
            if current_section:
                chunks.append('\n'.join(current_section))
            current_section = [text]
        else:
            current_section.append(text)

    if current_section:
        chunks.append('\n'.join(current_section))

    return chunks

def is_section_start(text):
    """
    Detecta se texto é início de seção
    """
    # Exemplo: linhas que começam com número seguido de ponto
    import re
    return bool(re.match(r'^\d+\.', text.strip()))
```

**Quando usar**: Documentos com estrutura clara (código, markdown, HTML)

### 5. Token-Based Chunking
Divide baseado em contagem de tokens (mais preciso).

```python
import tiktoken

def token_based_chunking(text, model="gpt-3.5-turbo", max_tokens=512, overlap_tokens=50):
    """
    Chunk baseado em tokens (não caracteres)
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - overlap_tokens

    return chunks

# Uso
chunks = token_based_chunking(
    text,
    model="gpt-3.5-turbo",
    max_tokens=512,
    overlap_tokens=50
)
```

**Vantagens**:
- Preciso para limites de LLM
- Consistente entre modelos
- Evita truncamento

**Desvantagens**:
- Mais lento
- Depende de tokenizer específico

**Quando usar**: Quando precisão de token count é crítica

## Overlap Entre Chunks

### Por que Usar Overlap?

```python
# Sem overlap
chunks = ["A B C", "D E F", "G H I"]
# Perde contexto na transição

# Com overlap
chunks = ["A B C", "C D E", "E F G", "G H I"]
# Mantém contexto na transição
```

### Quanto Overlap?

**Regra Geral**: 10-20% do chunk size

```python
chunk_size = 512
overlap = int(chunk_size * 0.15)  # 15% = 76 caracteres
```

**Trade-offs**:
- **Mais overlap**: Melhor contexto, mais chunks (mais custo, mais armazenamento)
- **Menos overlap**: Menos chunks, perde contexto nas bordas

### Exemplo do Exame
```
Pergunta: 150M embeddings, database suporta 100M. Como reduzir?
Resposta:
A. Aumentar chunk size (menos chunks)
B. Diminuir overlap (menos duplicação)
```

## Otimização de Chunking

### Processo de Otimização

```python
def optimize_chunking(documents, test_queries):
    """
    Testa diferentes configurações de chunking
    """
    configurations = [
        {"size": 256, "overlap": 25},
        {"size": 512, "overlap": 50},
        {"size": 1024, "overlap": 100},
    ]

    results = []

    for config in configurations:
        # 1. Fazer chunking
        chunks = chunk_documents(documents, **config)

        # 2. Criar embeddings
        embeddings = create_embeddings(chunks)

        # 3. Testar retrieval
        metrics = evaluate_retrieval(embeddings, test_queries)

        results.append({
            "config": config,
            "num_chunks": len(chunks),
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "avg_chunk_size": metrics['avg_size']
        })

    # Selecionar melhor
    best = max(results, key=lambda x: x['recall'])
    return best
```

### Métricas para Avaliar Chunking

1. **Retrieval Precision/Recall**: Chunks corretos recuperados?
2. **Chunk Size Distribution**: Tamanhos muito variados?
3. **Coverage**: Informação preservada?
4. **Storage**: Quanto espaço ocupam embeddings?

## Casos Específicos

### Caso 1: Documentos Técnicos com Código
```python
# Estratégia: Manter funções/classes intactas
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,  # Maior para não quebrar funções
    chunk_overlap=100
)
```

### Caso 2: FAQs (Pergunta-Resposta)
```python
# Estratégia: Cada Q&A é um chunk
def chunk_faq(faq_text):
    """
    Divide FAQs em pares Q&A individuais
    """
    chunks = []
    entries = faq_text.split('\n\nQ:')

    for entry in entries:
        if entry.strip():
            chunks.append('Q:' + entry.strip())

    return chunks
```

### Caso 3: Artigos Longos
```python
# Estratégia: Chunks por seção com introdução repetida
def chunk_article_with_intro(article):
    """
    Cada chunk inclui introdução + seção
    """
    intro = extract_introduction(article)
    sections = extract_sections(article)

    chunks = []
    for section in sections:
        chunk = f"{intro}\n\n{section}"
        chunks.append(chunk)

    return chunks
```

### Caso 4: Tabelas
```python
# Estratégia: Manter tabela inteira OU dividir por linha mantendo header
def chunk_table(table_text):
    """
    Divide tabela grande mantendo header em cada chunk
    """
    lines = table_text.split('\n')
    header = lines[0]
    rows = lines[1:]

    chunks = []
    for i in range(0, len(rows), 10):  # 10 linhas por chunk
        chunk_rows = rows[i:i+10]
        chunk = '\n'.join([header] + chunk_rows)
        chunks.append(chunk)

    return chunks
```

## Metadata em Chunks

Adicione metadata para melhorar retrieval:

```python
def chunk_with_metadata(document):
    """
    Cria chunks com metadata
    """
    chunks = []

    for i, chunk_text in enumerate(split_document(document)):
        chunk = {
            'text': chunk_text,
            'metadata': {
                'source': document['source'],
                'doc_id': document['id'],
                'chunk_id': i,
                'total_chunks': len(split_document(document)),
                'section': detect_section(chunk_text),
                'type': document['type']  # 'faq', 'manual', 'code', etc.
            }
        }
        chunks.append(chunk)

    return chunks
```

**Benefícios**:
- Filtering durante busca
- Ordenação por relevância
- Rastreabilidade (qual doc original?)

## Best Practices

### ✅ Faça
1. **Teste múltiplas configurações**: Não use primeira tentativa
2. **Use overlap**: 10-20% do chunk size
3. **Respeite estrutura**: Quando possível
4. **Adicione metadata**: Source, chunk_id, tipo
5. **Monitore métricas**: Precision/recall de retrieval
6. **Considere seu modelo**: Context length do embedding

### ❌ Evite
1. **Chunks muito pequenos**: < 100 caracteres perde contexto
2. **Chunks muito grandes**: > context length do modelo
3. **Quebrar no meio de frases**: Use semantic boundaries
4. **Overlap excessivo**: > 30% desperdiça recursos
5. **Ignorar tipo de documento**: One size doesn't fit all

## Checklist de Decisão

Ao escolher estratégia de chunking, pergunte:

- [ ] Qual o context length do meu embedding model?
- [ ] Que tipo de documentos tenho? (narrativo, técnico, código)
- [ ] Que tipo de queries espero? (específicas, gerais)
- [ ] Tenho limitações de armazenamento?
- [ ] Preciso otimizar custo ou qualidade?
- [ ] Documentos têm estrutura clara?

## Questões de Revisão

1. Por que usar overlap entre chunks?
2. Quando aumentar o chunk size reduz número de embeddings?
3. Qual estratégia usar para código Python?
4. Quanto overlap é recomendado (% do chunk size)?
5. Como otimizar chunking baseado em métricas de retrieval?

## Exercícios Práticos

### Exercício 1
Implemente e compare 3 estratégias:
- Fixed-size (512 chars, overlap 50)
- Semantic (por parágrafo)
- Recursive (LangChain)

Meça: número de chunks, tamanho médio, desvio padrão

### Exercício 2
Dado: 150M embeddings, database suporta 100M
Calcule: qual combinação de chunk_size e overlap atinge objetivo?

### Exercício 3
Crie estratégia customizada para documentação técnica com:
- Código Python
- Texto explicativo
- Tabelas

## Recursos Adicionais

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Chunking Strategies Guide](https://www.pinecone.io/learn/chunking-strategies/)

---

[← Anterior: Prompt Engineering](./01-prompt-engineering.md) | [Próximo: Vector Search →](./03-vector-search.md)
