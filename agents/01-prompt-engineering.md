# Agente: Prompt Engineering

## Objetivo
Dominar técnicas de prompt engineering para criar prompts efetivos que geram respostas estruturadas e controladas de LLMs.

## Conceitos Fundamentais

### O que é Prompt Engineering?
Prompt engineering é a arte e ciência de criar instruções (prompts) que direcionam LLMs a produzir outputs desejados com alta qualidade e consistência.

### Por que é Importante?
- Controla formato e estrutura da resposta
- Reduz alucinações
- Melhora relevância e precisão
- Otimiza custos (prompts melhores = menos tentativas)

## Técnicas Principais

### 1. Zero-Shot Prompting
Instrução direta sem exemplos.

```python
prompt = """
Classifique o sentimento da seguinte avaliação como positivo, negativo ou neutro:

Avaliação: {review}

Sentimento:
"""
```

**Quando usar**: Tarefas simples, modelos grandes e capazes.

### 2. Few-Shot Prompting
Fornecer exemplos de entrada/saída esperada.

```python
prompt = """
Classifique o sentimento das avaliações:

Avaliação: "Produto excelente, recomendo!"
Sentimento: positivo

Avaliação: "Chegou quebrado, péssimo."
Sentimento: negativo

Avaliação: "Produto ok, nada especial."
Sentimento: neutro

Avaliação: {review}
Sentimento:
"""
```

**Quando usar**: Tarefas específicas, formato customizado, modelos menores.

### 3. Chain-of-Thought (CoT)
Instruir o modelo a mostrar raciocínio passo a passo.

```python
prompt = """
Responda a pergunta mostrando seu raciocínio passo a passo:

Pergunta: Se um produto custa R$100 e tem 20% de desconto,
depois tem mais 10% de desconto, qual o preço final?

Pensemos passo a passo:
1. Preço inicial: R$100
2. Primeiro desconto de 20%: R$100 * 0.8 = R$80
3. Segundo desconto de 10% sobre R$80: R$80 * 0.9 = R$72
Resposta: R$72

Agora responda:
Pergunta: {question}
Pensemos passo a passo:
"""
```

**Quando usar**: Raciocínio complexo, cálculos, problemas multi-etapas.

### 4. Role Prompting
Atribuir um papel/expertise ao modelo.

```python
prompt = """
Você é um especialista em atendimento ao cliente com 10 anos de experiência.
Seu objetivo é resolver problemas de forma empática e eficiente.

Situação: {situation}

Como você responderia?
"""
```

**Quando usar**: Respostas especializadas, tom específico, contexto profissional.

### 5. Structured Output Prompting
Garantir formato específico na resposta.

```python
prompt = """
Analise o texto e retorne APENAS um JSON válido no formato:
{
  "sentimento": "positivo/negativo/neutro",
  "topicos": ["topico1", "topico2"],
  "prioridade": "alta/media/baixa",
  "confianca": 0.0-1.0
}

Texto: {text}

JSON:
"""
```

**Quando usar**: Integração com sistemas, parsing automático, dados estruturados.

### 6. Negative Prompting
Especificar o que NÃO fazer.

```python
prompt = """
Responda a pergunta de forma concisa.

REGRAS:
- NÃO invente informações
- NÃO use mais de 3 frases
- NÃO inclua exemplos
- NÃO mencione informações não solicitadas

Pergunta: {question}

Resposta:
"""
```

**Quando usar**: Evitar comportamentos indesejados, controlar verbosidade.

## Prompts Anti-Hallucination

### Técnica 1: Limitar ao Contexto
```python
prompt = """
Você é um assistente que responde APENAS baseado em informações fornecidas.

REGRAS IMPORTANTES:
1. Se a informação NÃO está no contexto, diga "Não tenho informação suficiente"
2. NUNCA invente fatos, datas, números ou detalhes
3. Se incerto, expresse a incerteza claramente
4. Cite a fonte quando possível

Contexto:
{context}

Pergunta: {question}

Resposta baseada APENAS no contexto:
"""
```

### Técnica 2: Grounded Generation
```python
prompt = """
Para cada afirmação na sua resposta:
1. Ela deve ter suporte direto nos documentos
2. Inclua [ref: doc_id] após cada afirmação

Documentos:
{documents}

Pergunta: {question}

Resposta com referências:
"""
```

### Técnica 3: Uncertainty Expression
```python
prompt = """
Ao responder:
- Use "Eu sei que..." para informações certas
- Use "Eu acredito que..." para inferências
- Use "Eu não tenho certeza sobre..." para incertezas
- Diga "Não sei" quando não souber

Pergunta: {question}

Resposta:
"""
```

## Prompts para Proteção de Privacidade

```python
privacy_prompt = """
Você é um assistente que protege informações privadas.

REGRAS DE PRIVACIDADE:
1. NUNCA revele: emails, telefones, CPF, senhas, endereços completos
2. Se perguntado sobre dados privados, responda genericamente
3. Substitua dados sensíveis por marcadores: [EMAIL], [TELEFONE], etc.
4. Não memorize ou repita informações privadas de conversas anteriores

{conversation}

Responda de forma útil MAS protegendo privacidade:
"""
```

## Delimitadores e Formatação

### Uso de Delimitadores
```python
prompt = """
Analise o texto entre aspas triplas e resuma em uma frase.

Texto: \"\"\"
{long_text}
\"\"\"

Resumo em uma frase:
"""
```

**Delimitadores comuns**:
- `"""` - Aspas triplas
- `###` - Hash triplo
- `---` - Traços
- `<text>...</text>` - Tags XML
- `[TEXT]...[/TEXT]` - Tags customizadas

### Instruções de Sistema vs. User Input
```python
# Separação clara
system_prompt = """
Você é um assistente especializado em produtos eletrônicos.
Sempre responda de forma técnica e precisa.
"""

user_template = """
Pergunta do usuário: {user_input}

Resposta:
"""
```

## Template com LangChain

```python
from langchain import PromptTemplate

# Template simples
template = PromptTemplate(
    input_variables=["produto", "tom"],
    template="""
    Escreva uma descrição {tom} para o seguinte produto.

    Produto: {produto}

    Descrição:
    """
)

# Usar template
prompt = template.format(produto="Smartphone 5G", tom="entusiasmado")
```

### Chat Prompt Template
```python
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

system_template = "Você é um assistente especializado em {area}."
system_message = SystemMessagePromptTemplate.from_template(system_template)

human_template = "{pergunta}"
human_message = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([
    system_message,
    human_message
])

# Usar
messages = chat_prompt.format_messages(
    area="tecnologia",
    pergunta="O que é 5G?"
)
```

## Best Practices

### ✅ Faça
1. **Seja específico**: "Resuma em 3 frases" vs. "Resuma"
2. **Use exemplos**: Few-shot quando formato é complexo
3. **Defina formato**: JSON, lista, tabela, etc.
4. **Estabeleça limites**: Tamanho, escopo, restrições
5. **Teste iterativamente**: Refine baseado em resultados
6. **Use delimitadores**: Separe instruções de conteúdo

### ❌ Evite
1. **Ambiguidade**: "Explique isso" - isso o quê?
2. **Prompts muito longos**: LLMs perdem foco
3. **Instruções conflitantes**: "Seja breve" e "Detalhe tudo"
4. **Assumir contexto**: Forneça todas informações necessárias
5. **Ignorar temperatura**: Ajuste para tarefa (criativa vs. factual)

## Exemplos por Caso de Uso

### Caso 1: Extração de Informações
```python
prompt = """
Extraia as seguintes informações do texto:
- Nome do produto
- Preço
- Categoria
- Avaliação (1-5 estrelas)

Retorne em formato JSON.

Texto: {text}

JSON:
"""
```

### Caso 2: Sumarização
```python
prompt = """
Resuma o seguinte artigo em exatamente 3 bullet points.
Cada bullet point deve ter no máximo 20 palavras.
Foque nos pontos principais e insights.

Artigo:
{article}

Resumo (3 bullets):
-
"""
```

### Caso 3: Classificação
```python
prompt = """
Classifique o ticket de suporte nas seguintes categorias:
- URGENTE: Problema crítico, sistema fora do ar
- ALTA: Funcionalidade importante não funciona
- MEDIA: Bug menor, workaround disponível
- BAIXA: Dúvida, melhoria, feature request

Ticket: {ticket_description}

Categoria:
"""
```

### Caso 4: Geração de Código
```python
prompt = """
Gere código Python que:
1. {requirement_1}
2. {requirement_2}
3. {requirement_3}

Requisitos:
- Use type hints
- Inclua docstrings
- Trate erros apropriadamente
- Siga PEP 8

Código Python:
"""
```

## Questões de Revisão

1. Qual a diferença entre zero-shot e few-shot prompting?
2. Quando usar Chain-of-Thought prompting?
3. Como estruturar um prompt que retorna JSON?
4. Que técnicas reduzem alucinações?
5. Por que usar delimitadores em prompts?

## Exercícios Práticos

### Exercício 1
Crie um prompt que:
- Extrai nome, email e telefone de texto livre
- Retorna JSON estruturado
- Valida formato dos dados
- Indica confiança da extração

### Exercício 2
Crie um prompt anti-hallucination para RAG que:
- Limita resposta ao contexto fornecido
- Inclui referências às fontes
- Expressa incerteza quando apropriado

### Exercício 3
Compare zero-shot vs. few-shot para classificação de sentimento.
Teste com 10 exemplos e meça precisão.

## Recursos Adicionais

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)
- [LangChain Prompt Templates](https://python.langchain.com/docs/modules/model_io/prompts/)

---

[← Voltar aos Agentes](../agents/) | [Próximo: Chunking Strategies →](./02-chunking-strategies.md)
