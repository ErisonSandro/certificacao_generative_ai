# Agentes de Estudo - Guias Focados

Esta pasta contém guias focados em tópicos específicos da certificação. Cada agente é um guia auto-contido sobre um tópico importante.

## 🎯 Master Agent - Comece Aqui!

**[00-master-agent.md](./00-master-agent.md)** - O agente inteligente que direciona você para o agente correto baseado na sua pergunta!

**Use o Master Agent quando**:
- Não souber qual agente consultar
- Tiver uma pergunta complexa que envolve múltiplos tópicos
- Quiser entender como os agentes se relacionam
- Precisar de um direcionamento rápido

---

## Como Usar

Cada agente pode ser estudado independentemente, mas a ordem sugerida é:

1. **Comece pelo Master Agent** para entender o ecossistema
2. **Fundamentos** (Agentes 1-2)
3. **Dados e Retrieval** (Agentes 2-3)
4. **Desenvolvimento** (Agentes 4, 7)
5. **Deploy e Produção** (Agentes 5-6)

## Lista de Agentes

### 🎯 Agente 1: Prompt Engineering
**Arquivo**: [01-prompt-engineering.md](./01-prompt-engineering.md)

**O que você vai aprender**:
- Técnicas de prompt engineering (zero-shot, few-shot, CoT)
- Prompts anti-hallucination
- Structured output prompting
- Metaprompts para proteção de privacidade
- Best practices

**Peso no exame**: ⭐⭐⭐⭐⭐ (Muito importante!)

**Quando estudar**: Semana 1 - É fundamental!

---

### 📦 Agente 2: Chunking Strategies
**Arquivo**: [02-chunking-strategies.md](./02-chunking-strategies.md)

**O que você vai aprender**:
- Estratégias de chunking (fixed-size, semantic, recursive)
- Como overlap funciona
- Otimização baseada em métricas
- Document-specific chunking
- Trade-offs e decisões

**Peso no exame**: ⭐⭐⭐⭐⭐ (Muito cobrado!)

**Quando estudar**: Semana 3 - Essencial para RAG

---

### 🔍 Agente 3: Vector Search
**Arquivo**: [03-vector-search.md](./03-vector-search.md)

**O que você vai aprender**:
- Databricks Vector Search endpoints e indices
- Delta Sync vs Direct Vector Access
- Criação e consulta de indices
- Similarity search com filtros
- Integração com LangChain
- Otimização de performance

**Peso no exame**: ⭐⭐⭐⭐⭐ (Muito importante!)

**Quando estudar**: Semana 8 - Específico do Databricks

---

### 🛡️ Agente 4: Guardrails e Segurança
**Arquivo**: [04-guardrails.md](./04-guardrails.md)

**O que você vai aprender**:
- Input guardrails (prompt injection detection)
- Output guardrails (toxicity, PII detection)
- PII masking techniques
- Rate limiting
- Metaprompts de segurança
- Guardrails AI library

**Peso no exame**: ⭐⭐⭐⭐ (Importante!)

**Quando estudar**: Semana 6 - Critical para produção

---

### 🚀 Agente 5: Model Serving e Unity Catalog
**Arquivo**: [05-model-serving-unity-catalog.md](./05-model-serving-unity-catalog.md)

**O que você vai aprender**:
- Criar Model Serving endpoints
- Registrar modelos no Unity Catalog
- PyFunc models
- Permissões e governança
- Batch inference com ai_query()

**Peso no exame**: ⭐⭐⭐⭐⭐ (Muito importante!)

**Quando estudar**: Semana 8-9 - Deploy

---

### 📊 Agente 6: MLflow e Evaluation
**Arquivo**: [06-mlflow-evaluation.md](./06-mlflow-evaluation.md)

**O que você vai aprender**:
- mlflow.evaluate() para RAG
- Métricas de qualidade (BLEU, ROUGE)
- Métricas customizadas
- Inference logging
- Monitoramento de custos
- LLM-as-judge

**Peso no exame**: ⭐⭐⭐ (Moderado)

**Quando estudar**: Semana 11 - Evaluation

---

### ⛓️ Agente 7: LangChain Basics
**Arquivo**: [07-langchain-basics.md](./07-langchain-basics.md)

**O que você vai aprender**:
- Chains (simple, sequential, RAG)
- Agents e tools
- Memory (buffer, window)
- Retrievers
- Output parsers

**Peso no exame**: ⭐⭐⭐⭐ (Importante!)

**Quando estudar**: Semana 5-7 - Development

---

## Mapa de Tópicos por Seção do Exame

### Seção 1: Design Applications (18%)
- 🎯 Agente 1: Prompt Engineering
- ⛓️ Agente 7: LangChain Basics

### Seção 2: Data Preparation (22%)
- 📦 Agente 2: Chunking Strategies
- 🔍 Agente 3: Vector Search (parte de retrieval)

### Seção 3: Application Development (27%) ⭐ Maior seção!
- 🎯 Agente 1: Prompt Engineering
- 🛡️ Agente 4: Guardrails
- ⛓️ Agente 7: LangChain Basics

### Seção 4: Assembling and Deploying (22%)
- 🔍 Agente 3: Vector Search
- 🚀 Agente 5: Model Serving e Unity Catalog

### Seção 5: Governance (7%)
- 🛡️ Agente 4: Guardrails (PII, security)

### Seção 6: Evaluation and Monitoring (4%)
- 📊 Agente 6: MLflow e Evaluation

---

## Ordem de Estudo Sugerida

### Para Iniciantes
1. Agente 1 - Prompt Engineering
2. Agente 7 - LangChain Basics
3. Agente 2 - Chunking Strategies
4. Agente 3 - Vector Search
5. Agente 4 - Guardrails
6. Agente 5 - Model Serving
7. Agente 6 - MLflow

### Para Quem Tem Experiência
Foque nos específicos do Databricks:
1. Agente 3 - Vector Search ⭐
2. Agente 5 - Model Serving ⭐
3. Agente 2 - Chunking (revisar otimização)
4. Agente 4 - Guardrails (compliance)
5. Agente 6 - MLflow

### Revisão Pre-Exame (1 semana)
Priorize os ⭐⭐⭐⭐⭐:
1. Agente 1 - Prompt Engineering
2. Agente 2 - Chunking
3. Agente 3 - Vector Search
4. Agente 5 - Model Serving

---

## Como Estudar Cada Agente

1. **Leia Completo** (~30-60 min)
2. **Teste os Códigos** - Execute os exemplos
3. **Faça os Exercícios** - Pratique hands-on
4. **Responda Questões** - Auto-avaliação
5. **Crie Resumo** - Anote pontos principais

---

## Checklist de Domínio

Marque quando dominar cada agente:

- [ ] Agente 1: Prompt Engineering
- [ ] Agente 2: Chunking Strategies
- [ ] Agente 3: Vector Search
- [ ] Agente 4: Guardrails
- [ ] Agente 5: Model Serving e Unity Catalog
- [ ] Agente 6: MLflow e Evaluation
- [ ] Agente 7: LangChain Basics

---

## Integrações Entre Agentes

Muitos conceitos se conectam:

```
Agente 2 (Chunking)
    ↓
Agente 3 (Vector Search)  ←→  Agente 7 (LangChain)
    ↓
Agente 1 (Prompts) + Agente 4 (Guardrails)
    ↓
Agente 5 (Model Serving)
    ↓
Agente 6 (Evaluation)
```

---

## Comandos Rápidos

### Para Databricks
```python
# Vector Search
from databricks.vector_search.client import VectorSearchClient
client = VectorSearchClient()

# Model Serving
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# MLflow
import mlflow
mlflow.set_registry_uri("databricks-uc")
```

### Para LangChain
```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import DatabricksVectorSearch
from langchain.llms import Databricks
```

---

## Recursos Adicionais

Além destes agentes, consulte também:
- [README principal](../README.md) - Visão geral da certificação
- [Questões de exemplo](../questoes-exemplo.md) - 20 questões práticas
- [Guia de preparação](../guia-preparacao.md) - Plano de 12 semanas
- Seções completas (pastas 01-06) - Conteúdo detalhado

---

## Contribuindo

Se você:
- Encontrou erros nos agentes
- Tem sugestões de melhorias
- Quer adicionar mais exemplos

Sinta-se livre para contribuir!

---

## Última Atualização

Este material cobre a versão do exame vigente desde Abril 2025.

**Dica**: Verifique atualizações 2 semanas antes do seu exame!

---

**Boa sorte nos estudos!** 🚀

Lembre-se: Estes agentes são guias focados. Para conteúdo completo, consulte as seções principais (pastas 01-06).

---

[← Voltar ao README Principal](../README.md)
