import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
import transformers
from transformers import Qwen2Config


if __name__ == "__main__":
    n_num=20
    i_curr=0
    vocab = {
        "<unk>": i_curr,
        "<bos>": i_curr+1,
        "<pad>": i_curr + 2,
        ":": i_curr + 3,
        "S": i_curr + 4,
        "P": i_curr + 5,
        "C": i_curr + 6,
        "A": i_curr + 7,
        "I": i_curr + 8,
    }
    i_curr += len(vocab)
    for i in range(n_num):
        vocab[f"{i}"] = i_curr
        i_curr += 1

    model_name=f"cfpark00/toy-multistep-v5"

    tokenizer_model = WordLevel(vocab=vocab, unk_token="<unk>")
    tokenizer = Tokenizer(tokenizer_model)
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,pad_token="<pad>",eos_token="<pad>")
    tokenizer.push_to_hub(model_name)

    model_config=Qwen2Config(
        hidden_size=128,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=len(tokenizer.vocab),
    )
    model=transformers.AutoModelForCausalLM.from_config(model_config)
    model.push_to_hub(model_name)