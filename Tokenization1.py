from transformers import AutoTokenizer

text = "کتابخانه‌های عمومی منابع ارزشمندی برای یادگیری هستند."

# مدل‌های مختلف توکن‌سازی chang1
tokenizers = {
    "BERT" : AutoTokenizer.from_pretrained("bert-base-multilingual-cased"),
    "GPT-2": AutoTokenizer.from_pretrained("gpt2")
}

# پردازش متن با توکن‌سازها
for model, tokenizer in tokenizers.items():
    tokens = tokenizer.tokenize(text)
    print(f"\n{model} Tokenization:")
    print(tokens)
