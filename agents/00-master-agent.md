# 🎯 Master Agent - Databricks Generative AI Certification

## 🤖 O que é o Master Agent?

O Master Agent é seu **guia inteligente** que:
- Interpreta sua pergunta e identifica o tópico
- Direciona você para o agente especializado correto
- Fornece resposta rápida com exemplo prático
- Alerta sobre pegadinhas comuns
- Conecta tópicos relacionados

**Como usar**: Faça sua pergunta naturalmente e o Master Agent identificará automaticamente o melhor caminho.

---

## 📊 Mapeamento de Tópicos

| Keywords Principais | Agente | Quando Usar | Dificuldade |
|-------------------|--------|-------------|-------------|
| prompt, few-shot, zero-shot, CoT | **01** Prompt Engineering | Melhorar qualidade de respostas | ⭐⭐⭐☆☆ |
| chunk, split, overlap, token | **02** Chunking | Preparar documentos para RAG | ⭐⭐⭐⭐☆ |
| vector, embedding, similarity, index | **03** Vector Search | Busca semântica e retrieval | ⭐⭐⭐⭐⭐ |
| PII, guardrails, security, injection | **04** Guardrails | Segurança e validação | ⭐⭐⭐☆☆ |
| endpoint, deploy, serving, Unity Catalog | **05** Model Serving | Deploy em produção | ⭐⭐⭐⭐⭐ |
| mlflow, metrics, evaluate, monitoring | **06** MLflow | Avaliação e monitoramento | ⭐⭐⭐☆☆ |
| langchain, chain, agent, tools, memory | **07** LangChain | Orquestração de workflows | ⭐⭐⭐⭐☆ |

---

## 📝 Formato de Resposta Padrão

Toda resposta do Master Agent segue este formato:

```
🎯 TÓPICO: [Nome do tópico]
📊 DIFICULDADE: ⭐⭐⭐☆☆ ([Básico/Intermediário/Avançado])
🤖 AGENTE: [Número e nome]


💡 RESPOSTA RÁPIDA:
[Explicação concisa]

🎨 Analogia:
[Analogia simples do dia-a-dia ]


🔨 EXEMPLO:
[Código ou conceito aplicado ]


⚠️ PEGADINHA:
❌ [O que NÃO fazer]
✅ [O que fazer corretamente]


📝 RESUMO:
[Ponto-chave para fixar]
```

---

## 💬 Exemplos de Uso

### Exemplo 1: Pergunta Simples

**Pergunta**: "Como fazer few-shot prompting?"

**Resposta**:
```
🎯 TÓPICO: Few-Shot Prompting
📊 DIFICULDADE: ⭐⭐⭐☆☆ (Intermediário)
🤖 AGENTE: 01 - Prompt Engineering

💡 RESPOSTA RÁPIDA:
Few-shot prompting fornece 2-5 exemplos no prompt para guiar o modelo.
Diferente de zero-shot (sem exemplos) e fine-tuning (retreinamento).

🎨 PENSE ASSIM:
É como mostrar fotos de exemplo antes de pedir para alguém desenhar algo.

🔨 EXEMPLO:
prompt = """
Classifique o sentimento:

Exemplos:
"produto excelente" → positivo
"produto horrível" → negativo

Classifique: {user_input}
"""

⚠️ PEGADINHA:
❌ Few-shot NÃO é fine-tuning (não muda pesos do modelo)
✅ Few-shot é aprendizado "in-context" - exemplos no prompt

🔗 RELACIONADO:
- Zero-shot prompting (Agente 01)
- Chain-of-Thought (Agente 01)
- RAG prompting patterns (Agente 01 + 03)

📝 RESUMO:
Few-shot = 2-5 exemplos no prompt. Não treina o modelo, apenas guia.
```



## 🎯 Quick Reference - Decisões Rápidas

| Sua Necessidade | Use Agente | Por quê |
|----------------|-----------|---------|
| Melhorar respostas do LLM | 01 | Técnicas de prompting |
| Preparar documentos | 02 | Chunking strategies |
| Fazer busca semântica | 03 | Vector search |
| Proteger dados sensíveis | 04 | Guardrails e PII |
| Deploy em produção | 05 | Model serving |
| Medir performance | 06 | Metrics e evaluation |
| Criar workflows complexos | 07 | LangChain chains |

---

## 📚 Todos os Agentes

### Agente 01: Prompt Engineering ⭐⭐⭐⭐⭐
**Peso no exame**: 27% (App Dev) + 18% (Design) = 45%
**Cobre**: Few-shot, zero-shot, CoT, structured outputs, anti-hallucination
**Quando usar**: Melhorar qualidade e controle das respostas do LLM

---

### Agente 02: Chunking Strategies ⭐⭐⭐⭐⭐
**Peso no exame**: 22% (Data Preparation)
**Cobre**: Fixed-size, semantic, recursive, chunk optimization, overlap
**Quando usar**: Preparar documentos para RAG, otimizar custos

---

### Agente 03: Vector Search ⭐⭐⭐⭐⭐
**Peso no exame**: 22% (Data Prep) + 22% (Assembly) = 44%
**Cobre**: Delta Sync, Direct Access, indices, similarity search, filtering
**Quando usar**: Implementar busca semântica e retrieval

---

### Agente 04: Guardrails ⭐⭐⭐⭐
**Peso no exame**: 7% (Governance) + parte de 27% (App Dev)
**Cobre**: PII detection, prompt injection, validation, toxicity filtering
**Quando usar**: Segurança, compliance e proteção de dados

---

### Agente 05: Model Serving & Unity Catalog ⭐⭐⭐⭐⭐
**Peso no exame**: 22% (Assembly and Deploying)
**Cobre**: Endpoints, PyFunc, Unity Catalog, workload sizing, ai_query
**Quando usar**: Deploy de modelos em produção

---

### Agente 06: MLflow & Evaluation ⭐⭐⭐
**Peso no exame**: 4% (Evaluation and Monitoring)
**Cobre**: Metrics (BLEU, ROUGE), LLM-as-judge, logging, cost tracking
**Quando usar**: Avaliar e monitorar performance

---

### Agente 07: LangChain Basics ⭐⭐⭐⭐
**Peso no exame**: 18% (Design) + 27% (App Dev) = 45%
**Cobre**: Chains, agents, tools, memory, RAG workflows
**Quando usar**: Orquestrar workflows complexos de LLM

---

## 🎓 Funcionalidades Especiais

### 1. Comando: Simulado

**Como usar**:
```
"Me dê questões do simulado sobre chunking"
"Simulado de prompt engineering"  
"Questões sobre vector search"
```

**O que faz**: Busca e apresenta questões relevantes do arquivo de simulado.

---

### 2. Busca Inteligente

Quando o Master Agent não souber algo:
1. **Primeiro**: Busca na documentação oficial do Databricks
2. **Depois**: Se não encontrar, busca na web
3. **Sempre**: Cita a fonte da informação

**Exemplo**:
```
🔍 Buscando em: Databricks Documentation...
✅ ENCONTRADO

[resposta]

📚 FONTE: https://docs.databricks.com/...
```


## 🔗 Mapa de Relacionamentos

```
        Agente 01 (Prompts)
              ↓
         [Qualidade]
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
Agente 07           Agente 02
(LangChain)         (Chunking)
    ↓                   ↓
[Workflow]         [Preparação]
    ↓                   ↓
    └─────────┬─────────┘
              ↓
        Agente 03
      (Vector Search)
              ↓
         [Retrieval]
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
Agente 04           Agente 05
(Guardrails)        (Serving)
    ↓                   ↓
[Segurança]         [Deploy]
    ↓                   ↓
    └─────────┬─────────┘
              ↓
        Agente 06
        (MLflow)
              ↓
      [Monitoramento]
```

---


## 🚀 Começando

**Para iniciantes**: 
1. Comece pelo Agente 01 (Prompts)
2. Depois Agente 02 (Chunking)
3. Depois Agente 03 (Vector Search)

**Para experientes**: 
Foque nos específicos do Databricks:
1. Agente 03 (Vector Search)
2. Agente 05 (Model Serving)
3. Agente 06 (MLflow)

**Revisão pré-exame**: 
Priorize os ⭐⭐⭐⭐⭐:
1. Agente 01, 02, 03, 05
2. Todas as questões do simulado
3. Pegadinhas comuns

---

## 📞 Como Pedir Ajuda ao Master Agent

### Exemplos de Comandos:

**Para aprender conceitos**:
- "O que é few-shot prompting?"
- "Como funciona vector search?"
- "Diferença entre Delta Sync e Direct Access"

**Para resolver problemas**:
- "Meu RAG está lento"
- "Como reduzir custos de embeddings"
- "Respostas do LLM estão ruins"

**Para praticar**:
- "Simulado sobre chunking"
- "Questões de prompt engineering"
- "Me dê questões difíceis"

**Para pesquisar**:
- "O que é ai_query no Databricks?"
- "Como funciona MLflow evaluate?"
- "Unity Catalog permissions"

---

## 🎯 Objetivos do Master Agent

1. **Interpretar** sua pergunta corretamente
2. **Classificar** o tópico em 1-2 segundos
3. **Responder** de forma objetiva e prática
4. **Conectar** tópicos relacionados
5. **Alertar** sobre pegadinhas comuns
6. **Facilitar** memorização com analogias
7. **Direcionar** para estudo aprofundado

---

