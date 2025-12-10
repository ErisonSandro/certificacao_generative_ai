# Agente: MLflow e Evaluation

## MLflow Evaluation

### Avaliar Modelo RAG

```python
import mlflow
import pandas as pd

# Dataset de teste
eval_data = pd.DataFrame({
    'query': [
        'Qual o preço do produto X?',
        'Como funciona a garantia?'
    ],
    'ground_truth': [
        'O produto X custa R$ 299,90',
        'Garantia de 12 meses'
    ]
})

# Avaliar
results = mlflow.evaluate(
    model=model_uri,
    data=eval_data,
    targets='ground_truth',
    model_type='text'
)

print(results.metrics)
```

### Métricas Customizadas

```python
def answer_relevance(eval_df, builtin_metrics):
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer('all-MiniLM-L6-v2')

    scores = []
    for _, row in eval_df.iterrows():
        q_emb = model.encode(row['query'], convert_to_tensor=True)
        a_emb = model.encode(row['outputs'], convert_to_tensor=True)
        similarity = util.cos_sim(q_emb, a_emb).item()
        scores.append(similarity)

    return {'answer_relevance': sum(scores) / len(scores)}

# Usar
results = mlflow.evaluate(
    model=model_uri,
    data=eval_data,
    targets='ground_truth',
    extra_metrics=[answer_relevance]
)
```

## Métricas Principais

### BLEU Score
```python
from nltk.translate.bleu_score import sentence_bleu

reference = "O produto chegou rápido".split()
hypothesis = "O item foi entregue rapidamente".split()

score = sentence_bleu([reference], hypothesis)
```

### ROUGE Score
```python
from rouge import Rouge

rouge = Rouge()

scores = rouge.get_scores(
    hypothesis="Produto bom",
    reference="Produto excelente"
)[0]

print(scores['rouge-l']['f'])
```

### LLM-as-Judge
```python
def llm_judge(query, response):
    judge_prompt = f"""
    Avalie de 1-5:
    Q: {query}
    A: {response}

    Critérios: relevância, completude, precisão

    Score:
    """

    score = judge_llm.invoke(judge_prompt)
    return extract_score(score)
```

## Inference Logging

### Habilitar

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

### Analisar Logs

```sql
-- Latência por hora
SELECT
  date_trunc('hour', timestamp) as hour,
  AVG(latency_ms) as avg_latency,
  COUNT(*) as requests
FROM catalog.schema.endpoint_request_logs
GROUP BY hour
ORDER BY hour;

-- Erros
SELECT *
FROM catalog.schema.endpoint_request_logs
WHERE status_code >= 400
ORDER BY timestamp DESC
LIMIT 100;
```

## Monitoramento de Custos

```python
class CostMonitor:
    def __init__(self):
        self.pricing = {
            'llama-2-7b': 0.0002,
            'llama-2-70b': 0.002
        }

    def log_usage(self, model, input_tokens, output_tokens):
        total_tokens = input_tokens + output_tokens
        cost = (total_tokens / 1000) * self.pricing[model]

        mlflow.log_metric('cost', cost)
        mlflow.log_metric('tokens', total_tokens)

        return cost
```

---

[← Anterior: Model Serving](./05-model-serving-unity-catalog.md) | [Próximo: LangChain →](./07-langchain-basics.md)
