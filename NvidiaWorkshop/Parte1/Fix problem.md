- List the models available through OpenAI client (not on the machine you're working on)
    

  

openai_client.models.list()

for model in models:

print(model.id)

  

- Replace the model variable in the formatting code block from
    

model = "mistralai/mistral-7b-instruct-v0.3"

model = "meta/llama3-70b-instruct"