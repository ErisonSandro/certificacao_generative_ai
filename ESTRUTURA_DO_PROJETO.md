# Estrutura do Projeto de Estudos

## Visao Geral

Este projeto foi criado para ajuda-lo a se preparar para a certificacao **Databricks Certified Generative AI Engineer Associate**.

## Estrutura de Pastas

```
certificacao_generative_ai/
│
├── README.md                                    # Visao geral da certificacao
├── guia-preparacao.md                          # Plano de estudos de 12 semanas
├── questoes-exemplo.md                         # 20 questoes praticas com respostas
├── ESTRUTURA_DO_PROJETO.md                     # Este arquivo
│
├── 01-design-applications/
│   └── README.md                               # Design de aplicacoes LLM
│
├── 02-data-preparation/
│   └── README.md                               # Preparacao de dados e chunking
│
├── 03-application-development/
│   └── README.md                               # Desenvolvimento de aplicacoes
│
├── 04-assembling-deploying/
│   └── README.md                               # Montagem e deploy
│
├── 05-governance/
│   └── README.md                               # Governanca e seguranca
│
└── 06-evaluation-monitoring/
    └── README.md                               # Avaliacao e monitoramento
```

## Como Usar Este Material

### 1. Para Iniciantes

Se voce esta comecando do zero:

1. Comece lendo [README.md](./README.md) para entender o exame
2. Leia [guia-preparacao.md](./guia-preparacao.md) e siga o plano de 12 semanas
3. Estude cada secao na ordem (1 a 6)
4. Pratique com as [questoes-exemplo.md](./questoes-exemplo.md)
5. Revise areas onde teve dificuldade

### 2. Para Quem Tem Experiencia

Se voce ja tem experiencia com LLMs/Databricks:

1. Leia [README.md](./README.md) rapidamente
2. Va direto para as secoes onde tem menos experiencia
3. Foque nas especificidades do Databricks (Vector Search, Unity Catalog, Model Serving)
4. Faca todas as [questoes-exemplo.md](./questoes-exemplo.md)
5. Use [guia-preparacao.md](./guia-preparacao.md) para areas especificas

### 3. Para Revisao Pre-Exame

Se seu exame e em breve:

1. Revise resumos de cada secao
2. Faca [questoes-exemplo.md](./questoes-exemplo.md) sem consultar
3. Revise questoes do PDF oficial
4. Foque em comandos e sintaxes especificas
5. Descanse bem antes do exame

## Conteudo de Cada Secao

### Secao 1: Design Applications
- Design de prompts estruturados
- Selecao de model tasks
- Componentes de chains
- Traducao de casos de uso
- Tools para agents

### Secao 2: Data Preparation
- Estrategias de chunking
- Extracao de documentos (PDF, imagens, HTML)
- Filtragem de conteudo
- Delta Lake e Unity Catalog
- Metricas de retrieval
- Re-ranking

### Secao 3: Application Development
- Prompt engineering avancado
- LangChain e ferramentas similares
- Guardrails (input e output)
- Metaprompts anti-hallucination
- Selecao de LLMs e embedding models
- Agent frameworks

### Secao 4: Assembling and Deploying
- PyFunc models
- Registro no Unity Catalog
- Model Serving endpoints
- Vector Search (Delta Sync e Direct Access)
- Foundation Model APIs
- Batch inference com ai_query()

### Secao 5: Governance
- PII masking e anonimizacao
- Protecao contra prompt injection
- Mitigacao de conteudo toxico
- LGPD/GDPR compliance
- Audit logging

### Secao 6: Evaluation and Monitoring
- Metricas de qualidade (BLEU, ROUGE, Perplexity)
- Metricas de performance (latencia, throughput)
- MLflow evaluation
- Inference logging
- Cost monitoring
- Agent monitoring
- LLM-as-judge

## Recursos por Secao

### Secao 1
- LangChain Quick Start
- Prompt Engineering Guide
- Databricks Academy courses

### Secao 2
- Delta Lake Guide
- Unity Catalog Documentation
- LangChain Text Splitters

### Secao 3
- Guardrails AI Documentation
- LangChain Agents
- Hugging Face Model Hub

### Secao 4
- MLflow Models Documentation
- Databricks Model Serving Guide
- Vector Search Documentation

### Secao 5
- Presidio Documentation
- LGPD text
- Unity Catalog Governance

### Secao 6
- MLflow Evaluation Guide
- Inference Tables Documentation
- Agent Monitoring Guide

## Estatisticas do Exame

- **Questoes**: 45 (multipla escolha e multipla selecao)
- **Tempo**: 90 minutos (~2 min por questao)
- **Taxa de aprovacao**: Nao divulgada oficialmente
- **Custo**: $200 USD
- **Validade**: 2 anos
- **Formato**: Online supervisionado

## Distribuicao Aproximada de Questoes

Baseado no exam outline:

- **Design Applications**: ~8 questoes (18%)
- **Data Preparation**: ~10 questoes (22%)
- **Application Development**: ~12 questoes (27%)
- **Assembling and Deploying**: ~10 questoes (22%)
- **Governance**: ~3 questoes (7%)
- **Evaluation and Monitoring**: ~2 questoes (4%)

**Nota**: A secao 3 (Application Development) e a maior e mais importante!

## Topicos Mais Importantes

### Alto Peso (aparecem muito)
1. Estrategias de chunking
2. Prompt engineering
3. Vector Search (criacao e uso)
4. Model Serving (deploy)
5. Selecao de modelos LLM/embedding
6. Guardrails

### Medio Peso
1. PyFunc models
2. Unity Catalog (registro de modelos)
3. LangChain chains e agents
4. Extracao de documentos
5. Delta Lake operations
6. PII masking

### Baixo Peso (mas ainda podem aparecer)
1. Metricas especificas (BLEU, ROUGE)
2. Inference logging
3. Cost optimization
4. LGPD/GDPR detalhes
5. Re-ranking

## Dicas de Estudo

### Hands-On e Essencial
- Nao apenas leia, PRATIQUE!
- Configure Databricks Community Edition (gratis)
- Implemente cada exemplo de codigo
- Crie seus proprios projetos

### Foque em Databricks
- Muitas questoes sao sobre ferramentas Databricks especificas
- Vector Search, Unity Catalog, Model Serving
- Sintaxe e comandos especificos

### Entenda Conceitos, Nao Decore
- Entenda QUANDO usar cada tecnica
- Entenda TRADE-OFFS de cada abordagem
- Saiba comparar alternativas

### Questoes do PDF Oficial
- As 5 questoes do PDF sao MUITO similares ao exame
- Entenda nao so a resposta, mas o PORQUE
- Questoes de exemplo deste projeto seguem o mesmo estilo

## Proximos Passos

1. [ ] Ler README.md completo
2. [ ] Escolher plano de estudos (12 semanas ou customizado)
3. [ ] Configurar ambiente Databricks
4. [ ] Comecar Secao 1
5. [ ] Fazer anotacoes pessoais
6. [ ] Praticar codigo
7. [ ] Fazer questoes de exemplo
8. [ ] Agendar exame quando estiver pronto

## Atualizacoes

Este material cobre a versao do exame vigente desde **18 de abril de 2025**.

**IMPORTANTE**: Verifique no site oficial da Databricks se ha atualizacoes 2 semanas antes do seu exame.

## Feedback e Melhorias

Se voce:
- Encontrou erros
- Tem sugestoes de melhoria
- Quer adicionar conteudo
- Passou no exame e quer compartilhar dicas

Sinta-se a vontade para contribuir!

## Licenca e Uso

Este material e para fins educacionais. Use para seus estudos, compartilhe com outros estudantes, mas respeite os direitos autorais dos materiais referenciados.

---

**Boa sorte nos seus estudos!**

A jornada para a certificacao e uma oportunidade de aprender profundamente sobre Generative AI e Databricks. Aproveite o processo!

---

Criado com base no Databricks Exam Guide oficial.
Ultima atualizacao: 2025
