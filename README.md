# Databricks Certified Generative AI Engineer Associate - Guia de Estudos

## 🚨 PROVA EM 4 DIAS?

**[→ CLIQUE AQUI para o Plano Intensivo de 4 Dias ←](./PREPARACAO-4-DIAS.md)**

Plano emergencial focado nos tópicos de maior peso no exame!

---

## Sobre a Certificacao

A certificacao **Databricks Certified Generative AI Engineer Associate** avalia a capacidade de um individuo em projetar e implementar solucoes habilitadas por LLM usando Databricks.

### Informacoes do Exame

- **Numero de questoes pontuadas**: 45 questoes (multipla escolha ou multipla selecao)
- **Tempo limite**: 90 minutos
- **Taxa de inscricao**: $200
- **Metodo de entrega**: Online Proctored (supervisionado online)
- **Materiais de apoio**: Nenhum permitido
- **Pre-requisitos**: Nenhum obrigatorio (recomendado: curso + 6 meses de experiencia pratica)
- **Validade**: 2 anos
- **Recertificacao**: Necessaria a cada 2 anos

### O que voce vai aprender

Esta certificacao cobre:
- Decomposicao de problemas complexos em tarefas gerenciaveis
- Escolha de modelos, ferramentas e abordagens apropriadas do cenario atual de IA generativa
- Ferramentas especificas do Databricks:
  - **Vector Search**: busca por similaridade semantica
  - **Model Serving**: implantacao de modelos e solucoes
  - **MLflow**: gerenciamento do ciclo de vida da solucao
  - **Unity Catalog**: governanca de dados

Individuos que passam neste exame podem construir e implantar aplicacoes RAG performaticas e chains de LLM que aproveitam totalmente o Databricks e seu conjunto de ferramentas.

## Estrutura do Exame

### [Secao 1: Design Applications](./01-design-applications/README.md)
- Design de prompts com formatacao especifica
- Selecao de tarefas de modelo
- Selecao de componentes de chain
- Traducao de casos de uso de negocio
- Definicao de ferramentas para raciocinio multi-estagio

### [Secao 2: Data Preparation](./02-data-preparation/README.md)
- Estrategias de chunking
- Filtragem de conteudo
- Extracao de documentos
- Operacoes com Delta Lake e Unity Catalog
- Identificacao de documentos fonte
- Avaliacao de desempenho de recuperacao

### [Secao 3: Application Development](./03-application-development/README.md)
- Criacao de ferramentas para extracao de dados
- Uso de LangChain e ferramentas similares
- Prompt engineering
- Avaliacao qualitativa de respostas
- Implementacao de guardrails
- Selecao de modelos LLM e embedding
- Desenvolvimento de sistemas agentivos

### [Secao 4: Assembling and Deploying Applications](./04-assembling-deploying/README.md)
- Codificacao de chains usando pyfunc
- Controle de acesso a recursos
- Criacao de chains simples e com LangChain
- Registro de modelos no Unity Catalog
- Implantacao de endpoints
- Vector Search (criacao e consulta)
- Inference em lote com ai_query()

### [Secao 5: Governance](./05-governance/README.md)
- Tecnicas de mascaramento como guardrails
- Protecao contra entradas maliciosas
- Mitigacao de texto problematico
- Requisitos legais e de licenciamento
- Conformidade e seguranca

### [Secao 6: Evaluation and Monitoring](./06-evaluation-monitoring/README.md)
- Selecao de LLM baseada em metricas quantitativas
- Metricas-chave para monitoramento
- Avaliacao de desempenho usando MLflow
- Inference logging
- Controle de custos de LLM
- Inference tables e Agent Monitoring
- Comparacao entre fases de avaliacao e monitoramento

## Preparacao Recomendada

### Cursos Databricks Academy
- **ILT Course**: Generative AI Engineering with Databricks
- **Self-Paced**: Generative AI Engineering with Databricks

#### Novos Modulos (em breve):
1. Generative AI Solution Development (RAG)
2. Generative AI Application Development (Agents)
3. Generative AI Application Evaluation and Governance
4. Generative AI Application Deployment and Monitoring

### Conhecimentos Necessarios
- LLMs atuais e suas capacidades
- Prompt engineering, geracao e avaliacao de prompts
- Ferramentas online: LangChain, Hugging Face Transformers
- Python e bibliotecas para desenvolvimento de aplicacoes RAG e chains LLM
- APIs para preparacao de dados, encadeamento de modelos
- Documentacao relevante do Databricks

## Recursos de Estudo

### 📚 Guias Focados por Tópico (NOVO!)

#### 🎯 Master Agent - Comece Aqui!
**[Master Agent](./agents/00-master-agent.md)** - Agente inteligente que identifica sua pergunta e direciona para o agente especializado correto!

#### Agentes Especializados
Confira a pasta [agents/](./agents/) com guias focados em tópicos específicos:
- **Prompt Engineering** - Técnicas e best practices
- **Chunking Strategies** - Otimização de divisão de documentos
- **Vector Search** - Databricks Vector Search completo
- **Guardrails** - Segurança e validações
- **Model Serving** - Deploy e Unity Catalog
- **MLflow** - Evaluation e monitoramento
- **LangChain** - Chains, agents e memory

👉 [Ver todos os agentes](./agents/README.md)

### Documentacao Oficial
- [Databricks Documentation](https://docs.databricks.com/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Unity Catalog Documentation](https://docs.databricks.com/data-governance/unity-catalog/index.html)

### Questoes de Exemplo
Confira [questoes-exemplo.md](./questoes-exemplo.md) para questoes praticas similares ao exame real.

## Como Usar Este Guia

1. **Estude cada secao**: Navegue pelas pastas de cada secao e estude o conteudo
2. **Pratique**: Implemente os exemplos e exercicios propostos
3. **Revise**: Use as questoes de exemplo para testar seu conhecimento
4. **Repita**: Revisite areas onde precisa de mais pratica

## Progresso de Estudos

Use a lista abaixo para acompanhar seu progresso:

- [ ] Secao 1: Design Applications
- [ ] Secao 2: Data Preparation
- [ ] Secao 3: Application Development
- [ ] Secao 4: Assembling and Deploying Applications
- [ ] Secao 5: Governance
- [ ] Secao 6: Evaluation and Monitoring
- [ ] Questoes de Exemplo
- [ ] Revisao Final

## Dicas para o Exame

1. **Leia atentamente**: Algumas questoes tem multiplas respostas corretas
2. **Gerencie o tempo**: 90 minutos para 45 questoes = ~2 minutos por questao
3. **Nao deixe em branco**: Nao ha penalizacao para respostas incorretas
4. **Marque para revisao**: Use a funcao de marcar questoes para revisar depois
5. **Foque no Databricks**: Muitas questoes sao especificas das ferramentas Databricks

## Atualizacoes

Este guia cobre a versao do exame vigente desde **18 de abril de 2025**. Verifique sempre 2 semanas antes do seu exame se ha atualizacoes.

---

Boa sorte nos seus estudos!
