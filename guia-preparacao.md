# Guia de Preparacao para a Certificacao

## Plano de Estudos Sugerido

### Semanas 1-2: Fundamentos e Design Applications

#### Objetivos
- Entender arquitetura de aplicacoes LLM
- Dominar prompt engineering
- Aprender sobre chains e agents

#### Atividades
1. **Estudo Teorico** (10 horas)
   - Ler [Secao 1: Design Applications](./01-design-applications/README.md)
   - Estudar LangChain documentation
   - Assistir Databricks Academy course

2. **Pratica** (10 horas)
   - Implementar chains simples
   - Criar prompts estruturados
   - Testar different chain types
   - Implementar agent basico com tools

3. **Revisao** (2 horas)
   - Fazer questoes de exemplo da Secao 1
   - Revisar conceitos que teve dificuldade

#### Recursos
- [LangChain Quick Start](https://python.langchain.com/docs/get_started/quickstart)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- Databricks Academy: Generative AI Engineering with Databricks

---

### Semanas 3-4: Data Preparation

#### Objetivos
- Dominar estrategias de chunking
- Aprender extracao de documentos
- Entender Delta Lake e Unity Catalog

#### Atividades
1. **Estudo Teorico** (8 horas)
   - Ler [Secao 2: Data Preparation](./02-data-preparation/README.md)
   - Estudar Delta Lake documentation
   - Revisar Unity Catalog concepts

2. **Pratica Hands-On** (12 horas)
   - Implementar diferentes estrategias de chunking
   - Extrair texto de PDFs, imagens, HTML
   - Criar pipeline completo de preparacao de dados
   - Escrever dados em Delta Tables
   - Testar diferentes tamanhos de chunk

3. **Projetos**
   - Projeto 1: Pipeline de ETL para documentos RAG
   - Projeto 2: Comparar estrategias de chunking

4. **Revisao** (2 horas)
   - Questoes de exemplo
   - Revisar metricas de retrieval

#### Recursos
- [Delta Lake Guide](https://docs.databricks.com/delta/index.html)
- [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)

---

### Semanas 5-7: Application Development

#### Objetivos
- Dominar prompt engineering avancado
- Implementar guardrails
- Selecionar modelos apropriados
- Criar agents complexos

#### Atividades
1. **Estudo Teorico** (12 horas)
   - Ler [Secao 3: Application Development](./03-application-development/README.md)
   - Estudar Guardrails AI
   - Aprender sobre model selection

2. **Pratica Intensiva** (18 horas)
   - Criar prompts anti-hallucination
   - Implementar input/output guardrails
   - Testar diferentes LLMs
   - Implementar RAG application completa
   - Criar agent com multiplos tools
   - Implementar memory em chains

3. **Projetos**
   - Projeto 1: Chatbot com guardrails completos
   - Projeto 2: Agent com tool calling
   - Projeto 3: Comparacao de LLMs

4. **Revisao** (3 horas)
   - Questoes de exemplo
   - Revisar conceitos complexos

#### Recursos
- [Guardrails AI](https://docs.guardrailsai.com/)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [Hugging Face Model Hub](https://huggingface.co/models)

---

### Semanas 8-9: Assembling and Deploying

#### Objetivos
- Criar pyfunc models
- Registrar modelos no Unity Catalog
- Deploy de Model Serving endpoints
- Dominar Vector Search

#### Atividades
1. **Estudo Teorico** (10 horas)
   - Ler [Secao 4: Assembling and Deploying](./04-assembling-deploying/README.md)
   - Estudar MLflow documentation
   - Aprender Vector Search

2. **Pratica em Databricks** (15 horas)
   - Criar pyfunc model customizado
   - Registrar modelo no Unity Catalog
   - Criar Vector Search index
   - Deploy de Model Serving endpoint
   - Testar endpoint via API
   - Configurar permissoes

3. **Projetos**
   - Projeto 1: RAG completo com Vector Search
   - Projeto 2: Deploy end-to-end de aplicacao

4. **Revisao** (2 horas)
   - Questoes de exemplo
   - Praticar comandos de deploy

#### Recursos
- [MLflow Models](https://mlflow.org/docs/latest/models.html)
- [Databricks Model Serving](https://docs.databricks.com/machine-learning/model-serving/index.html)
- [Vector Search Guide](https://docs.databricks.com/generative-ai/vector-search.html)

---

### Semana 10: Governance

#### Objetivos
- Implementar PII masking
- Proteger contra prompt injection
- Entender compliance (LGPD/GDPR)

#### Atividades
1. **Estudo Teorico** (6 horas)
   - Ler [Secao 5: Governance](./05-governance/README.md)
   - Estudar LGPD/GDPR requirements
   - Aprender sobre Presidio

2. **Pratica** (8 horas)
   - Implementar PII detection e masking
   - Criar sistema de guardrails completo
   - Implementar audit logging
   - Testar deteccao de prompt injection

3. **Projetos**
   - Projeto: Sistema de governanca completo

4. **Revisao** (2 horas)
   - Questoes de exemplo
   - Revisar best practices

#### Recursos
- [Presidio](https://microsoft.github.io/presidio/)
- [LGPD](http://www.planalto.gov.br/ccivil_03/_ato2015-2018/2018/lei/l13709.htm)
- [Unity Catalog Governance](https://docs.databricks.com/data-governance/unity-catalog/index.html)

---

### Semana 11: Evaluation and Monitoring

#### Objetivos
- Avaliar modelos com metricas
- Configurar inference logging
- Monitorar custos
- Usar MLflow evaluate

#### Atividades
1. **Estudo Teorico** (6 horas)
   - Ler [Secao 6: Evaluation and Monitoring](./06-evaluation-monitoring/README.md)
   - Estudar MLflow evaluation
   - Aprender sobre inference tables

2. **Pratica** (10 horas)
   - Calcular BLEU, ROUGE scores
   - Configurar inference logging
   - Implementar cost monitoring
   - Usar mlflow.evaluate()
   - Criar dashboard de metricas

3. **Projetos**
   - Projeto: Framework de evaluation completo

4. **Revisao** (2 horas)
   - Questoes de exemplo
   - Revisar metricas

#### Recursos
- [MLflow Evaluation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
- [Inference Tables](https://docs.databricks.com/machine-learning/model-serving/inference-tables.html)
- [Agent Monitoring](https://docs.databricks.com/generative-ai/agent-monitoring/index.html)

---

### Semana 12: Revisao Final e Simulados

#### Objetivos
- Revisar todos os topicos
- Fazer simulados
- Identificar pontos fracos

#### Atividades
1. **Revisao Geral** (10 horas)
   - Revisar anotacoes de todas as secoes
   - Fazer resumos dos conceitos principais
   - Revisar questoes que errou

2. **Simulados** (8 horas)
   - Fazer [questoes-exemplo.md](./questoes-exemplo.md) sem consultar
   - Fazer questoes do PDF oficial
   - Simular condicoes de exame (90 min, 45 questoes)

3. **Foco nos Pontos Fracos** (6 horas)
   - Identificar topicos com mais erros
   - Re-estudar esses topicos
   - Praticar mais

4. **Preparacao Final** (2 horas)
   - Revisar comandos principais
   - Revisar formulas e metricas
   - Descansar bem antes do exame

---

## Checklist de Preparacao

### Conhecimentos Fundamentais
- [ ] Entendo arquitetura de RAG applications
- [ ] Sei criar prompts efetivos
- [ ] Conheço diferentes tipos de chains
- [ ] Sei quando usar agents vs. chains

### Data Preparation
- [ ] Domino estrategias de chunking
- [ ] Sei extrair texto de diferentes formatos
- [ ] Conheço bibliotecas de extracao (pytesseract, PyPDF2, etc.)
- [ ] Sei escrever dados em Delta Lake
- [ ] Entendo metricas de retrieval

### Application Development
- [ ] Domino prompt engineering avancado
- [ ] Sei implementar guardrails
- [ ] Conheço tecnicas anti-hallucination
- [ ] Sei selecionar LLMs baseado em requisitos
- [ ] Sei selecionar embedding models
- [ ] Domino LangChain

### Assembling and Deploying
- [ ] Sei criar pyfunc models
- [ ] Sei registrar modelos no Unity Catalog
- [ ] Sei criar Vector Search indices
- [ ] Sei fazer deploy de Model Serving endpoints
- [ ] Entendo diferenca entre Delta Sync e Direct Vector Access
- [ ] Sei usar ai_query() para batch inference

### Governance
- [ ] Sei implementar PII masking
- [ ] Sei detectar prompt injection
- [ ] Conheço requisitos de LGPD/GDPR
- [ ] Sei implementar audit logging
- [ ] Conheço best practices de seguranca

### Evaluation and Monitoring
- [ ] Sei calcular metricas de qualidade (BLEU, ROUGE)
- [ ] Sei usar mlflow.evaluate()
- [ ] Sei configurar inference logging
- [ ] Sei monitorar custos
- [ ] Entendo diferenca entre evaluation e monitoring
- [ ] Sei criar dashboards de metricas

---

## Recursos Consolidados

### Documentacao Oficial Databricks
- [Databricks Generative AI](https://docs.databricks.com/generative-ai/index.html)
- [MLflow](https://mlflow.org/docs/latest/index.html)
- [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html)
- [Model Serving](https://docs.databricks.com/machine-learning/model-serving/index.html)
- [Vector Search](https://docs.databricks.com/generative-ai/vector-search.html)

### Frameworks e Bibliotecas
- [LangChain Documentation](https://python.langchain.com/)
- [Hugging Face](https://huggingface.co/)
- [Guardrails AI](https://docs.guardrailsai.com/)
- [Presidio](https://microsoft.github.io/presidio/)

### Cursos e Treinamentos
- Databricks Academy: Generative AI Engineering with Databricks
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [DeepLearning.AI - LangChain Course](https://www.deeplearning.ai/)

### Pratica
- [Databricks Community Edition](https://community.cloud.databricks.com/) (free tier)
- Criar projetos pessoais de RAG
- Participar de competitions no Kaggle

---

## Dicas para o Dia do Exame

### Antes do Exame
1. Durma bem na noite anterior
2. Chegue cedo (ou conecte cedo se online)
3. Tenha agua por perto
4. Verifique camera e internet (exame online)
5. Revise resumos rapidos, nao estude conteudo novo

### Durante o Exame
1. **Gerencie o tempo**: 90 min / 45 questoes = 2 min por questao
2. **Leia com atencao**: Identifique palavras-chave (DUAS acoes, MAIS efetivo, NAO apropriado)
3. **Elimine opcoes erradas**: Ajuda a focar nas certas
4. **Marque para revisao**: Questoes dificeis, volte depois
5. **Nao deixe em branco**: Nao ha penalizacao por erro
6. **Confie no seu preparo**: Se estudou bem, vai passar!

### Tipos de Questoes
- **Multipla escolha**: Uma resposta correta
- **Multipla selecao**: Duas ou mais respostas corretas (sempre especificado)
- **Cenario**: Descricao de situacao + pergunta
- **Codigo**: Podem mostrar codigo e perguntar sobre ele

### Gestao de Tempo Sugerida
- **Primeira passagem** (60 min): Responder todas que souber
- **Segunda passagem** (20 min): Resolver marcadas para revisao
- **Terceira passagem** (10 min): Revisar respostas, verificar erros bobos

---

## Apos o Exame

### Se Passar
- Celebre! Voce e Databricks Certified Generative AI Engineer Associate!
- Adicione ao LinkedIn
- Mantenha-se atualizado (certificacao valida por 2 anos)
- Compartilhe conhecimento com a comunidade

### Se Nao Passar
- Nao desanime! Muitos precisam de mais de uma tentativa
- Revise areas onde teve dificuldade
- Aguarde periodo de espera para refazer (verifique regras)
- Estude mais e tente novamente

---

## Comunidade e Suporte

### Grupos e Forums
- [Databricks Community Forum](https://community.databricks.com/)
- Stack Overflow (tag: databricks)
- Reddit: r/databricks, r/MachineLearning
- Discord/Slack communities sobre ML/AI

### Mantenha-se Atualizado
- [Databricks Blog](https://www.databricks.com/blog)
- [Databricks YouTube Channel](https://www.youtube.com/c/Databricks)
- Newsletter da Databricks
- Seguir #databricks no Twitter/LinkedIn

---

## Checklist Final (1 dia antes do exame)

- [ ] Revisei todas as 6 secoes
- [ ] Fiz todas as questoes de exemplo
- [ ] Entendo os conceitos principais de cada secao
- [ ] Sei comandos basicos de MLflow, LangChain, Vector Search
- [ ] Revisei as 5 questoes do PDF oficial
- [ ] Testei camera e internet (se online)
- [ ] Sei o horario e local do exame
- [ ] Estou descansado e confiante

---

**Boa sorte no seu exame!**

Lembre-se: A certificacao valida seu conhecimento, mas o verdadeiro valor esta no que voce aprendeu e nas aplicacoes que voce sera capaz de construir!

---

[← Voltar ao README Principal](./README.md)
