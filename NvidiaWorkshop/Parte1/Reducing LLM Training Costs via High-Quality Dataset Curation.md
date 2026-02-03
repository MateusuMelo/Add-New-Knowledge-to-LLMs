Para treinamentos de gigantescos datasets extraídos da internet é necessário garantir a qualidade dos dados passando por diversas filtragens. Para lidar com varias destas etapas de gerenciamento dos dados o curso oferece a ferramenta NeMo Curator


# NeMo Curator

## 1. Download

NeMo consegue realizar download dos dados das plataformas [Common Crawl](https://commoncrawl.org/) , [Wikipedia](https://dumps.wikimedia.org/****) e [ArXiv](https://info.arxiv.org/help/bulk_data_s3.html) para artigos científicos.

## 2. Processing

Formatar e limpar os dados coletados, removendo metadados das aplicações host (html metadados) que não agrega com informações relevantes.
## 3. Deduplication

### Fuzzy
Ao invés de ser exato, remover os dados que são parecidos em sentidos parecidos, usando MinHash
### Exact
Remover dados exatos baseado em hash

### Semantic
Remover duplicações que contem o mesmo sentido semântico 


## 4. Filtragem

### Filtragem por Classificação
### Filtragem por qualidade

### Filtragem por Heurística


#  Quality Filter



