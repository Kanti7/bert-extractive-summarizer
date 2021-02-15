from summarizer import Summarizer
from transformers import AutoConfig, AutoTokenizer, AutoModel

test_file = 'tests/test_news_source.txt'
model_spec = "distilbert-base-uncased"

f = open(test_file, 'r')
body = f.read()
f.close()

custom_config = AutoConfig.from_pretrained(model_spec)
custom_config.output_hidden_states=True
custom_tokenizer = AutoTokenizer.from_pretrained(model_spec)
custom_model = AutoModel.from_pretrained(model_spec, config=custom_config)
model = Summarizer(custom_model=custom_model, return_list=True, return_index=True)
result, index = model(body)

print('\n'.join(result))
print(index)