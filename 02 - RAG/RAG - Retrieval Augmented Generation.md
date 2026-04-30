Ao usar RAG injetamos as informações necessárias no prompt para responder as questões iniciais do usuário. Apos isso passamos o prompt aumentado para LLM para geração da resposta final. Assim, a LLM pode usar o contexto adicional para solucionar os problemas de alucinação e informações desatualizadas ou faltantes. 

## Funcionamento Do Rag Comum

Sistemas RAG são compostos por três módulos principais:

- **Ingestion Pipeline**: Uma pipeline do tipo batch ou streaming usada para popular o vector DB 
- **Retrieval Pipeline**: O modulo que consulta o vector DB e extrai informações relevantes a entrada do usuário.
- **Generation Pipeline**: A camada que usa os dados da camada Retrieval para enriquecer o prompt para que a LLM gere respostas. 

### Ingestion pipeline

Esta pipeline tem a responsabilidade de extrair os documentos brutos de varias fontes (ex, data warehouse, data lake, paginas web, etc). Então, limpa-os, divide em chunks e aplica embedding.

Assim, Ingestion pipelines são divididas em:

#### Data extraction module

O modulo que pega todos os dados necessarios de diversos tipos de bancos de dados, APIs ou paginas web. Sua montagem e totalmente dependente dos dados que são esperados, pode ser algo simples como uma query em um banco de dados ou mais complexo como crawling Wikipidia.

#### cleaning layer

Padroniza e remove caracteres não esperados dos dados. Por exemplo, removendo todos caracteres invalidos de um texto, como non-ASCII, negrito, italico. Troca URLs com placeholders. A estrategia de limpeza dos dados varia dependendo da fonte de dados e do metodo de embedding. 

#### Chunking module

Divide os dados limpos em pedaços menores, para realizar embedding, e necessário para garantir que a entrada do modelo não ultrapasse seu tamanho máximo. Também, e necessário separar regiões especificas que são semanticamente parecidas. Por exemplo, ao realizar chunking em um capitulo de livro é ideal manter os parágrafos parecidos na mesma seção ou no mesmo chunk. 


#### Embedding component

Usa um modelo de embedding para transformar os dados divididos (chunks) como textos, imagens, audios em vetores densos ajustados com seu sentido semântico. 


#### Loading module

Pega os vetores resultantes do embedding e insere no banco vetorial. Os metadados são parte principal desta etapa, e ideal adicionar informações essenciais como o que é o conteúdo, a URL da fonte do chunk, e quando o conteúdo foi disponibilizado. Pode ser util na hora da consulta pois os metadados ajudam a filtrar em casos em que somente a busca vetorial não gera resultados satisfatórios.

### Retrieval Pipeline

Consiste em pegar a entrada do usuário (texto, imagem, audio), realizar embedding e depois consultar o banco de dados vetoriais para achar vetores similares com a entrada do usuário.

A função retrieval deve realizar o embedding da entrada do usuário para o mesmo espaço latente do banco vetorial. Isso permite realizar a busca dos top $K$s entradas mais similares comparando o vetor do usuário com os do banco vetorial. 

As principais métricas para comparar vetores são a distancia Euclidiana e Manhattan. Porém a mais popular é a distancia de Cosseno.

$$
Cosine Distance = 1 - \cos(\theta) = 1 \frac{A \cdot B}{\| A \ \| \cdot
\| B \|
 }
$$

Com amplitude de -1 a 1, com -1 sendo a divergência total dos vetores, 0 com nenhuma correlação e 1 com a mesma correlação. 

O ponto crucial é que os dados de input e do banco vetorial precisam estar no mesmo espaço vetorial, isso implica em ter que passar todas as funções e parametros de processamento da criação do banco vetorial na entrada do usuário para garantir que não haja erros de busca.


### Generation pipeline

Ultima etapa, responsável por pegar os dados coletados junto com a entrada do usuário e gerar o prompt final para LLM. Dependendo da aplicação é necessário ter um ou mais prompt templates aplicando técnicas de prompt engineering.

```python
system_template = """ You are a helpful assistant who answers all the user's questions politely. """
 prompt_template = """ Answer the user's question using only the provided context. If you cannot answer using the context, respond with "I don't know." Context: {context} User question: {user_question} """ 
user_question = "<your_question>" 
retrieved_context = retrieve(user_question)
prompt = f"{system_template}\n"
prompt+=prompt_template.format(context=retrieved_context,user_question=user_question) 
answer = llm(prompt)

```

Ao aplicar prompt templates é necessário trackear e versioná-los aplicando as boas praticas MLOps. E possivel fazer isso através de Git, armazenar em bancos de dados, ou usar uma ferramenta especifica de gerenciamento de prompt como [LangFuse](https://langfuse.com/)

## Embedding

O melhor modelo  de embedding pode variar de acordo com o caso de uso. E possivel encontrar modelos particulares no [**Massive Text Embedding Benchmark (MTEB)**](https://huggingface.co/spaces/mteb/leaderboard) no Hugging Face. 

Quando se trata de dados em formato de audio, não podemos aplicar embedding comuns. Ao invés, precisamos criar uma representação ode visual do audio, como um espectrograma e por fim aplicar o modelo de embedding de imagem. 

Modelos como CLIP, podem praticamente juntar textos e imagens em um único espaço vetorial. Isso permite encontrar imagens similares usando uma sentença como entrada.

## Funcionamentos do RAG avançado

Para garantir uma maior qualidade nos dados coletados da RAG e respostar perguntas mais especificas como a relevância dos dados coletados para com a pergunta do usuário, se os dados coletados são suficientes para responder o usuário, o que fazer caso não seja possível gerar uma resposta valida para pergunta do usuário. Advanced RAG é projetada para responder todas estas perguntas e garantir a qualidade dos dados coletados.

O Rag comum pode ser otimizado acrescentando esses estagios:

### Pre-retrieval

Estagio focado em como estruturar e processar os dados para indexar os dados de uma maneira otimizada para melhorar consultas

pode ser performada de duas maneiras diferentes:

#### Data indexing:

É parte da pipeline de ingestão da Rag. É implementado principalmente nos módulos de limpeza ou fragmentação para pré-processar os dados para uma melhor indexação.

##### Sliding window

A técnica da janela deslizante introduz sobreposição entre trechos de texto, garantindo que o contexto importante próximo aos limites dos trechos seja mantido, o que aumenta a precisão da recuperação. Isso é particularmente benéfico em áreas como documentos jurídicos, artigos científicos, registros de atendimento ao cliente e registros médicos, onde informações críticas geralmente abrangem várias seções.

##### Enhancing data granularity

Envolve técnicas de limpeza de dados removendo detalhes irrelevantes, verificando precisão factual, e atualizando informações datadas.

##### Metadata

Adicionar tags de metadados como datas, URLs, IDs externos ou marcadores de capítulos ajudam a filtrar os resultados. 


#### Query optimizations

Performa diretamente na consulta do usuário antes do embedding e consulta de chunks do banco vetorial.

##### Query routing

Baseado na entrada do Usuário, onde é necessário interagir com diferentes categoria de dados e requisitar cada categoria diferentemente. Roteamento de query é usado para decidir qual ação é tomada na entrada do Usuário, parecido com condições if/else. 
![[99 - Attachments/Images/Pasted image 20260203111540.png]]

##### Query rewriting

As vezes a entrada do usuario pode não alinhar perfeitamente com os dados estruturados. Query rewriting trata isso reformulando a pergunta do Usuário para uma melhor precisão na hora de buscar os dados. 

### Retrieval

Esta etapa pode ser otimizada de duas maneiras:

#### Improving embedding model

Em alguns casos e necessário que o modelo de embedding também seja especialista em alguns tipos de jargões ou domínios, para isso é necessário realizar fine-tunning nos modelos de embeddings pré treinados.

A alternativa de realizar fine-tunning é usar modelos instrutores.

#### Leveraging DBs filter and search features

Uma das opções é a busca hibrida que combina busca semântica com busca baseada em palavras. Muitas vezes somente a busca semântica pode não entregar resultados específicos sendo necessária a aplicação de buscas por palavras chave. Ha o parâmetro comumente chamado *alpha* que controla o peso entre os dois métodos 

Outro tipo é filtrar as buscas semânticas usando algum tipo de metadado como parametro. Por exemplo, buscando somente pesquisas de uma determinada URL, ou de um determinado assunto destacado nos metadados para assim depois aplicar a busca semântica ja nesses documentos pré filtrados. 

### Post-retrieval

Etapa inserida para otimizar os dados coletados para garantir que a LLM tenha uma boa performance sem problemas como janela de contexto limitada ou dados ruidosos. Contextos muito grandes ou com informações irrelevantes podem distrair a LLM. 

#### Re-ranking

Usar um cross-encoder para medir a similaridade da entrada do usuario com os dados coletados. Os dados são rankeados basedado em sua similaridade e somente o top N são mantidos para a LLM. 


![[99 - Attachments/Images/Pasted image 20260203133004.png]]