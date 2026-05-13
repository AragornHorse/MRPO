
def hh(tokenizer):
    def func(example):
        text = example['chosen']
        text = [t.split("\n\nAssistant:")[0] for t in text]
        text = [t.split("\n\nHuman:")[-1] for t in text]
        text = [tokenizer.apply_chat_template([
            {'role': "user", 'content': t},    
        ], tokenize=False, add_generation_prompt=True) for t in text]
        return {"prompt": text}
    return func


def shp(example):
    return {"prompt": example['history']}


def uf(tokenizer):

    def func(example):
        text = [tokenizer.apply_chat_template([
             {'role': "user", 'content': t},
         ], tokenize=False, add_generation_prompt=True) for t in example['prompt']]
        return {"prompt": text}
    return func


def get_map(dataset, tokenizer):
    if dataset == "hh-rlhf":
        return hh(tokenizer)
    elif dataset == 'shp':
        return shp
    elif dataset == 'ultrafeedback':
        return uf(tokenizer)
    else:
        return None


def get_split(dataset):
    if dataset == 'ultrafeedback':
        return 'train_prefs'
    else:
        return 'train'


def get_drop_cols(dataset):
    if dataset == 'ultrafeedback':
        return ['messages', 'chosen', 'rejected']
    else:
        return None
