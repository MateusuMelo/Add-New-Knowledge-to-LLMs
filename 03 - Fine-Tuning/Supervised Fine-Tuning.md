
Esta etapa possui dois propósitos primário: ensina o modelo a entender e seguir um formato especifico de chat, transformando em um agente conversational, e permite que o modelo incremente seus conhecimentos para se destacar em tarefas especificas ou um domínio especifico. 

# SFT Techniques

SFT se consiste em retreinar um modelo pre-treinado em um dataset menor composto por pares de instruções e respostas. O objetivo do SFT e transformar um modelo base, que consegue somente predizer o próximo token, em um assistente util, capaz de responder perguntas e seguir instruções.

## Quando realizar fine-tune

Na grande maioria dos casos é recomendado começar com prompt engineering ao invés de já começar com fine-tuning. Prompt engineering pode ser usado tanto com modelos com pesos abertos (llama, qwen) ou restritos (gpt, gemini). Através das técnicas como few-shot prompting ou RAG, inúmeros problemas podem ser resolvidos sem a necessidade de SFT. Se mesmo assim os resultados não atingirem os requisitos, podemos explorar a necessidade de montar datasets de instruções. Se nenhum dado esta disponível, fine-tuning começa a ser uma opção viável. 

![[99 - Attachments/Images/Pasted image 20260122094046.png]]

SFT também possui limitações. Como SFT basicamente se aproveita do conhecimento pre-existente nos pesos do modelo base e reorienta os parâmetros para um objetivo especifico. Isso implica em algumas limitações, primeiro, se o conhecimento for muito distante daquele que foi aprendido no pre-treino (como uma linguagem rara ou desconhecida) pode ser difícil que o modelo entenda. 

## Formato do Dataset 

Datasets para SFT possuem o formato de Instrução, onde são organizados no formato de instrução e resposta. Os datasets podem ser representados em python dicts, onde as chaves são tipos de prompts como *system, instruction, output*. Os formatos mais conhecidos são *Alpaca*, *ShareGPT* e *OpenAI*.

| **Name**               | **JSONL format**                                                    |
| ---------------------- | ------------------------------------------------------------------- |
| **Alpaca**             | `{ "instruction": "...", "input": "...", "output": "..." }`         |
| **Alpaca (sem input)** | `{ "instruction": "...", "output": "..." }`                         |
| **ShareGPT**           | `{ "conversations": [ { "from": "...", "value": "..." }, ... ] }`   |
| **OpenAI**             | `{ "conversations": [ { "role": "...", "content": "..." }, ... ] }` |
| **OASST**              | `{ "INSTRUCTION": "...", "RESPONSE": "..." }`                       |
| **Raw text**           | `{ "text": "..." }`                                                 |


O formato Alpaca é suficiente quando precisamos de apenas uma instrução e resposta. Quando é necessário processar conversas  (múltiplas instruções e respostas), o formato ShareGPT ou OpenAI são melhores. 

## Chat Templates

Quando as instruções-respostas são extraídas do dataset, precisamos estruturá-las em Chat Template. Chat Template unificam a maneira como apresenta as instruções e respostas para o modelo.

Resumindo, eles incluem tokens especiais que identificam o começo e e o fim da mensagem, tao como os autores da mensagem. Modelos base não foram designados com chat template. Isso significa que podemos escolher qualquer template para realizar o fine-tuning. 

|**Name**|**Jinja template**|
|---|---|
|**Alpaca**|`### Instruction: What is the capital of France?\n### Response: The capital of France is Paris.<EOS>`|
|**ChatML**|`<\|im_start\|>user\nWhat is the capital of France?<\|im_end\|>\n<\|im_start\|>assistant\nThe capital of France is Paris.<\|im_end\|>`|
|**Llama 3**|`<\|begin_of_text\|><\|start_header_id\|>user<\|end_header_id\|>\nWhat is the capital of France?<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\nThe capital of France is Paris.<\|eot_id\|>`|
|**Phi-3**|`<\|user\|>\nWhat is the capital of France?<\|end\|>\n<\|assistant\|>\nThe capital of France is Paris.<\|end\|>`|
|**Gemma**|`<bos><start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\nThe capital of France is Paris.<end_of_turn>`|
# Fine-tuning Types
## Full Fine-tuning

Consiste em retreinar todo os parâmetros do modelo base. Como no pre-training, SFT usa a predição do próximo token como objetivo do treinamento. Sendo a principal diferença entre Full Fine-tuning sendo a estrutura do dataset explicada acima. 

Esta abordagem requisita um grande custo computacional, o uso de memoria utilizando uma unica GPU pode ser estimada resumidamente por:

$Memory = Parameters + Gradients + Optimizers States + Activations$

Para um setup usando fp32, podemos estimar:

- **Parameters**: Pesos e vieses ensináveis. Custo 4 bytes/Parametro (FP32) ou 2 bytes/Parâmetro (FP16/BF16).
- **Gradients**: 4 bytes/Parâmetro
- **Optimizer states**: valores gerenciados pelos algorítimos de otimização Adam ou AdamW. Custo: 8 bytes/Parâmetro

Para treinar uma LLM usando 16 bytes por parâmetro, no fim é necessário 112 GB de VRAM para um modelo de 7B e 1120 GB para um modelo de 70B.

Ha algumas técnicas usadas para reduzir o uso de memoria no fine-tuning. Paralelismo em distribui as tarefas em múltiplas gpus, em contra partida acrescenta sobrecarga. O acúmulo de gradientes permite tamanhos de lote efetivos maiores sem aumento proporcional da memória. Otimizadores eficientes em memoria como 8-bit adam. Checkpoints de ativação trocam o cálculo pela memória, recalculando determinadas ativações. Quando combinadas, estas tecnicas podem reduzir o custo de 16/bytes para 14-15/bytes por parametro.

Full fine-tuning modifica todos os pesos pre-treinados do modelo, o que por um lado pode ser destrutivo. Se o treinamento não ocorrer como esperado, pode acabar deletando a memoria anterior, um fenomeno chamado "Esquecimento Catastrófico". 

## LoRA

É uma tecnica de fine-tuning com eficiência em parâmetros. O propósito principal e permitir fine-tuning com redução em custo computacional. Isso se deve a aplicação da tecinica de matrizes de baixa dimensão que modifica  o comportamento do modelo sem alterar os pesos originais. As vantagens são :

- Redução drastica de uso de memoria no treinamento
- Fine-tuning mais rapido
- Preservar os pesos pre-treinados
- Habilidade de alternar entre as tarefas apenas mudando os pesos LoRA

O LoRA implementa uma técnica de decomposição em low-rank para atualizar os pesos eficientemente. Ao invés de modificar a matriz de pesos original $W$ , LoRA introduz duas matrizes menores, $A$ e $B$, que juntas formam uma atualização de low-rank.

![[99 - Attachments/Images/Pasted image 20260123093531.png]]

Matematicamente:

$W^{\prime} = W + BA$ 

Onde $W$ é a matriz original, $B,A$ as matrizes LoRA, e $W^{\prime}$ a matriz efetiva usada na inferência

O rank é definido por $r$ , que é crucial para LoRA. Para implementar efetivamente, precisamos selecionar corretamente os hiperparametros:

- **Rank ($r$)**: Determina o tamanho das matrizes LoRA. O ponto inicial é comumente $r=8$ , mas valores acima de 256 mostraram ser bons em alguns casos. Ranks maiores capturam tarefas mais diversas porem podem ocasionar em overfitting. 
- **Alpha ($\alpha$)**: Fator de escala aplicado a atualização do LoRA. Na pratica, atualizamos os pesos $W$ por um fator $\frac{\alpha}{r}$ . É por isso que uma heurística comum é definir $\alpha$  como o dobro do valor de $r$, aplicando efetivamente um fator de escala de 2 à atualização LoRA. Você pode experimentar diferentes proporções em caso de underfitting ou overfitting.

É possível acrescentar uma camada de dropout em para prevenir de overfitting. A taxa de dropout normalmente fica entre 0 e 0.1.

LoRA foi principalmente focada em modificar a região de atenção o, especificamente nas matrizes **query(Q)** e **value(V)** nas camadas transformer. Experimentos indicam beneficios quando aplicar LoRA em outros componentes:

- **Key(K)** : Matrizes na camada de atenção
- Camadas de projeção de saída (comumente representadas como O) nos mecanismos de atenção
- Blocos de Feed-forward ou MLP entre as camadas de atenção
- Camadas de saida linear

Usando LoRA, é possível realizar fine-tuning em modelos com 7B  de parâmetros usando uma unica GPU com 14-18 GB de Vram. 

Outra vantagem é que LoRA permite combinar diferentes tasks, aplicando o deploy de multiple-LoRA. 



## QLoRA

Em resumo é o LoRA aplicando também quantização nos parametros do modelo para o tipo NF4 (4-bit NormalFloat). Com o custo de incrementar o tempo de treinamento, aproximadamente 30% mais lento quando comparado ao LoRA.

# Training parameters

## Learning rate and scheduler

O hyper-parâmetro mais importante, responsável por definir o quanto o modelo ira atualizar seus parâmetros. Começa de valores bem pequenos $1e-6$  ate valores maiores $1e-3$.

## Batch Size

Determina o numero de amostras processadas antes do modelo atualizar seus pesos. Tipicamente batch sizes para fine-tuning de LLMs vão de 1 a 32. Quanto maiores , mais chances de alcançar gradientes estáveis garantindo uma velocidade maior de treinamento. Em contra partida aumenta o consumo de memoria.

==Para equilibrar o consumo de memoria e tamanho de batch existe a tecnica de ***acumulação de gradiente***.== Funciona passando mini-batches através da rede, acumulando os gradientes através dessas etapas antes de aplicar a atualização nos parâmetros do modelo.  Se seu batch size efetivo for de 32 mas a GPU somente consegue carregar 8 amostras ao mesmo tempo, a acumulação de gradiente pode ser configurada para 4 steps. Significa que voce processa 4 mini-batches de 8 amostras cada, resultando em um batch efetivo de 32. 

O numero de steps de acumulação de gradiente vai de 1 (sem acumulação) para 8 ou 16. A formula para definir o batch efetivo é

$Effective Batch Size = Batch Size \times GPUs\times Gradient Accumulation Steps$ 

Como exemplo, ao usar 2 GPUS, cada uma processando um batch de 4 amostras, com 4 steps de acumulação de gradiente, o batch efetivo é $4 \times 2 \times 4 = 32$ amostras


## Maximum length and packing

Define o tamanho máximo da entrada que o modelo pode processar. Geralmente fica entre 512 e 4096 tokens mas pode alcançar 128000 ou mais dependendo das capacidades da GPU. O tamanho padrão entre as linguagens tem em media 2048 tokens, enquanto modelos com RAG podem subir para 8192 ou mais. Este parâmetro impacta diretamente no batch size e no uso de memoria; um batch de 12 com um max length de 1024 ira conter 12288 tokens ($12 \times 1024$), enquanto para o mesmo tamanho de batch mas com max length de 512 ira conter 6144 tokens.

Packing maximiza a utilização de cada batch. Ao invés de atribuir uma unica amostra para o batch, packing combina múltiplas amostras menores em um único batch, aumentando a quantidade de dados processados a cada iteração. Por exemplo suponha que max length é de 1024 tokens, e a maioria das amostras dos datastes possuem 200-300 tokens, packing ajustara 3-4 samples para cada slot de batch.

## Epochs

Para fine-tuning de LLM o numero tipico de epocas varia entre 1 a 10, com diversas runs bem sucedidas com 2 a 5 épocas. Mais épocas permitem que o modelo refine o aprendizado, melhorando a performance. Poucas épocas podem ocasionar underfitting enquanto épocas demais podem ocasionar overfitting. Examinar o gráfico de validação pode ajudar implementando early stop no plateau do modelo. 

## Optimizer

Otimizadores ajustam os parâmetros do modelo para reduzir a função de loss. Para fine-tuning de LLM é recomendado a AdamW, particularmente a versão de 8-bits que é tem a performance parecida com a de 32-bit porem usando menos memoria.

## Weight decay

Uma técnica de regularização a decadência de peso funciona adicionando uma penalidade para pesos grandes à função de perda, incentivando o modelo a aprender características mais simples e generalizáveis. Configurar valores muito altos de decaimento de pesos pode impedir o modelo de aprender dificultando o modelo de aprender padrões dos dados.

## Gradient checkpointing

Técnica usada para reduzir o consumo de memoria durante o treinamento, armazenando somente um parte das ativações intermediarias geradas no forward pass. As ativações que não são salvas, são recalculadas ao passar o backward pass necessário para calculo do gradiente.



