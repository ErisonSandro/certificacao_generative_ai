# Agente: LangChain Essentials

## Chains

### Simple Chain
```python
from langchain.chains import LLMChain
from langchain import PromptTemplate

prompt = PromptTemplate(
    template="Descreva {produto}",
    input_variables=["produto"]
)

chain = LLMChain(llm=llm, prompt=prompt)
result = chain.run(produto="smartphone")
```

### Sequential Chain
```python
from langchain.chains import SequentialChain

chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="resumo")
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="analise")

seq_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["texto"],
    output_variables=["resumo", "analise"]
)
```

### RAG Chain
```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

result = qa_chain({"query": "pergunta"})
```

## Agents

### Create Agent
```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool

# Definir tools
tools = [
    Tool(
        name="Buscar",
        func=search_function,
        description="Busca informações"
    )
]

# Criar agent
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

result = agent_executor.invoke({"input": "pergunta"})
```

## Memory

### Conversation Buffer
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# Primeira interação
chain.run("Meu nome é João")

# Segunda interação (lembra o nome)
chain.run("Qual é meu nome?")
```

### Window Memory
```python
from langchain.memory import ConversationBufferWindowMemory

# Mantém apenas últimas 3 interações
memory = ConversationBufferWindowMemory(k=3)
```

## Retrievers

### Vector Store Retriever
```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

docs = retriever.get_relevant_documents("query")
```

### With Filters
```python
retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 5,
        "filter": {"category": "electronics"}
    }
)
```

## Output Parsers

### JSON Parser
```python
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="sentiment", description="positivo/negativo"),
    ResponseSchema(name="confidence", description="0-1")
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Adicionar ao prompt
format_instructions = parser.get_format_instructions()
```

---

[← Anterior: MLflow](./06-mlflow-evaluation.md) | [Voltar aos Agentes](./README.md)
