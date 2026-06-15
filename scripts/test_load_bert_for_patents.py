from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

model_name = "anferico/bert-for-patents"

print("Loading config...")
config = BertConfig.from_pretrained(
    model_name,
    num_labels=2,
)
print("Config loaded:", config.model_type)

print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained(model_name)
print("Tokenizer loaded:", type(tokenizer))

print("Loading model...")
model = BertForSequenceClassification.from_pretrained(
    model_name,
    config=config,
    ignore_mismatched_sizes=True,
)
print("Model loaded:", type(model))
