# Agente: Model Serving e Unity Catalog

## Model Serving

### Criar Endpoint

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ServedEntityInput, EndpointCoreConfigInput

w = WorkspaceClient()

# Configurar endpoint
served_entities = [
    ServedEntityInput(
        entity_name="catalog.schema.rag_model",
        entity_version="1",
        workload_size="Small",  # Small, Medium, Large
        scale_to_zero_enabled=True
    )
]

# Criar
endpoint = w.serving_endpoints.create(
    name="rag-endpoint",
    config=EndpointCoreConfigInput(served_entities=served_entities)
)
```

### Consultar Endpoint

```python
# Via SDK
response = w.serving_endpoints.query(
    name="rag-endpoint",
    inputs={"query": ["Pergunta teste"]}
)

# Via REST API
import requests

headers = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}

response = requests.post(
    f"{DATABRICKS_HOST}/serving-endpoints/rag-endpoint/invocations",
    headers=headers,
    json={"inputs": {"query": ["teste"]}}
)
```

### Atualizar Endpoint

```python
w.serving_endpoints.update_config(
    name="rag-endpoint",
    served_entities=[ServedEntityInput(
        entity_name="catalog.schema.rag_model",
        entity_version="2",  # Nova versão
        workload_size="Medium"
    )]
)
```

## Unity Catalog

### Registrar Modelo

```python
import mlflow

mlflow.set_registry_uri("databricks-uc")

model_name = "catalog.schema.rag_model"

with mlflow.start_run():
    mlflow.langchain.log_model(
        lc_model=chain,
        artifact_path="model",
        registered_model_name=model_name
    )
```

### Aliases e Versões

```python
client = mlflow.tracking.MlflowClient()

# Criar alias
client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version="1"
)

# Carregar por alias
model_uri = f"models:/{model_name}@champion"
model = mlflow.pyfunc.load_model(model_uri)
```

### Permissões

```python
w.grants.update(
    securable_type="FUNCTION",
    full_name="catalog.schema.rag_model",
    changes=[
        {"principal": "data-scientists", "add": ["EXECUTE"]},
        {"principal": "ml-engineers", "add": ["EXECUTE", "MODIFY"]}
    ]
)
```

## PyFunc Models

```python
import mlflow
from mlflow.pyfunc import PythonModel

class RAGModel(PythonModel):
    def load_context(self, context):
        # Carregar recursos
        self.llm = load_llm()
        self.retriever = load_retriever()

    def predict(self, context, model_input):
        query = model_input['query'][0]

        # RAG pipeline
        docs = self.retriever.get_relevant_documents(query)
        context_text = "\n".join([d.page_content for d in docs])
        prompt = f"Context: {context_text}\n\nQ: {query}\n\nA:"
        response = self.llm.invoke(prompt)

        return {"answer": response}

# Log
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=RAGModel(),
        pip_requirements=["langchain==0.1.0"]
    )
```

## Batch Inference com ai_query()

```sql
-- Processar tabela inteira
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

**Quando usar**:
- Batch processing
- ETL jobs
- Análises históricas

**Quando NÃO usar**:
- Real-time applications
- Latência crítica

---

[← Anterior: Guardrails](./04-guardrails.md) | [Próximo: MLflow →](./06-mlflow-evaluation.md)
