import matplotlib.cm as cm
import matplotlib.colors as mcolors
from IPython.display import display, HTML,clear_output
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
import copy
import torch
import numpy as np
import os
import json
import vllm
from vllm.lora.request import LoRARequest
from accelerate.utils import tqdm

def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def default_user_protocol(messages):
    user_input=input("Your message (\"END\" to quit): ")
    return {"role":"user","content":user_input,"done":user_input=="END"}

class LanguageModel():
    default_model_spec={
        "is_instruct": True,
        "has_system_prompt": True,
        "has_structured": False,
        }
    default_sampling_params={
        "temperature": 0.4,
        "n": 1,
    }
    def __init__(self,model_spec,model=None,**kwargs):
        self.model_spec=self.default_model_spec.copy()
        self.model_spec.update(model_spec)
        self.validate_spec()
        self.set_sampling_params()
        self.model=model
        if self.model is None:
            self.model=self.init_model(**kwargs)
    
    def validate_spec(self):
        assert "model_name" in self.model_spec, "model_name not found in model_spec"
        if self.model_spec["call_type"]=="hf":
            self.model_spec["max_new_tokens_key"]="max_completion_tokens"
        elif self.model_spec["call_type"]=="openai":
            self.model_spec["max_new_tokens_key"]="max_tokens"
        elif self.model_spec["call_type"]=="vllm":
            self.model_spec["max_new_tokens_key"]="max_tokens"
        else:
            raise Exception("Model call type not supported")

    def init_model(self,**kwargs):
        if self.model_spec["call_type"]=="hf":
            model= HFModel(self.model_spec)
            return model
        elif self.model_spec["call_type"]=="openai":
            model= OpenAIModel(self.model_spec)
            return model
        elif self.model_spec["call_type"]=="vllm":
            model= VLLMModel(self.model_spec,**kwargs)
            return model
        else:
            raise Exception("Model call type not supported")

    def set_sampling_params(self,**kwargs):
        self.sampling_params=self.default_sampling_params.copy()
        self.sampling_params.update(kwargs)

    def chat(self,messages,prefill_assistant=False,**kwargs):
        batched = isinstance(messages[0], list)
        if "user_protocol" not in kwargs:#single output
            return self.model.chat(self.get_raw_messages(messages),prefill_assistant=prefill_assistant,**self.sampling_params,**kwargs)
        elif kwargs["user_protocol"] is None:
            assert not batched, "batched conversation not supported with default user protocol, which required human input"
            user_protocol=default_user_protocol
        else:
            user_protocol=kwargs["user_protocol"]
        if prefill_assistant:
            assert messages[-1]["role"]=="assistant", "Last message should be assistant"
        #auto chat
        messages=copy.deepcopy(messages)
        if not batched:
            #prepare showing function
            show=kwargs.pop("show",True)
            cli_output=kwargs.pop("cli_output",False)
            if cli_output:
                def showfunc(messages,last_only=True):
                    if len(messages)==0:
                        return
                    if last_only:
                        print(messages_to_str([messages[-1]]))
                    else:
                        print(messages_to_str(messages))
            else:
                def showfunc(messages,last_only=True):
                    if len(messages)==0:
                        return
                    clear_output_and_show_messages(messages)
        
            user_turn=(len(messages)==0 or messages[-1]["role"]!="user") and not prefill_assistant
            showfunc(messages,last_only=False)
            while True:
                if user_turn:
                    message=user_protocol(messages)
                    messages.append(message)
                    showfunc(messages)
                    if "done" in message and message["done"]:
                        break
                else:
                    message=self.model.chat(self.get_raw_messages(messages),prefill_assistant=prefill_assistant,**self.sampling_params)
                    if prefill_assistant:
                        assert message["role"]=="assistant_completion", "Last message should be assistant_completion"
                        messages[-1]["content"]+=message["content"]
                    else:
                        messages.append(message)
                    showfunc(messages)
                user_turn=not user_turn
            return messages
        #batched chat
        if prefill_assistant:
            assert all([(len(messages_)>0 and messages_[-1]["role"]=="assistant") for messages_ in messages]), "Last message in batched conversation should be assistant if using prefill_assistant"
        else:
            assert all([(len(messages_)>0 and messages_[-1]["role"]=="user") for messages_ in messages]), "Last message in batched conversation should be user"
        user_turn=False
        todo=list(range(len(messages)))
        while True:
            if user_turn:
                todo_=todo.copy()
                for i in todo_:
                    message=user_protocol(messages[i])
                    messages[i].append(message)
                    if "done" in message and message["done"]:
                        todo.remove(i)
            else:
                #batched chat
                messages_todo=[]
                for i in todo:
                    messages_todo.append(messages[i])
                new_messages=self.model.chat(self.get_raw_messages(messages_todo),prefill_assistant=prefill_assistant,**self.sampling_params)
                assert len(new_messages)==len(todo), "Number of new messages should be same as number of todo"
                for i,new_message in zip(todo,new_messages):
                    if prefill_assistant:
                        assert new_message["role"]=="assistant_completion", "Last message should be assistant_completion"
                        messages[i][-1]["content"]+=new_message["content"]
                    else:
                        messages[i].append(new_message)
            if len(todo)==0:
                break
            user_turn=not user_turn
        return messages
    
    def get_raw_messages(self,messages):
        batched= isinstance(messages[0], list)
        if batched:
            raw_messages=[]
            for messages_ in messages:
                raw_messages_=[]
                for message in messages_:
                    raw_messages_.append({"role":message["role"],"content":message["content"]})
                raw_messages.append(raw_messages_)
            return raw_messages
        else:
            raw_messages=[]
            for message in messages:
                raw_messages.append({"role":message["role"],"content":message["content"]})
            return raw_messages
    
    def close(self):
        if self.model_spec["call_type"]=="vllm":
            free_vllm(self.model.model)

class HFModel():
    def __init__(self,model_spec):
        assert False, "Not checked"
        self.model_spec=model_spec
        self.model_name=model_spec["model_name"]
        self.model=AutoModelForCausalLM.from_pretrained(model_spec["model_name"])
        self.tokenizer=AutoTokenizer.from_pretrained(model_name,padding_side='left',device_map=model_spec.get("device_map","auto"),torch_dtype=model_spec.get("torch_dtype",torch.bfloat16))
        self.device=self.model.device
    
    def chat(self,messages,**kwargs):
        kwargs[self.model_spec["max_new_tokens_key"]]=kwargs.pop("max_new_tokens",4096)
        assert self.model_spec["is_instruct"], "Model is not an instruct model"
        if not self.model_spec["has_system_prompt"]:
            messages=system2user(messages)
        input_ids=self.tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors="pt")
        input_ids=input_ids.to(self.device)
        generation=self.generate(input_ids,**kwargs)
        response_text=self.tokenizer.decode(generation[0,input_ids.shape[1]:],skip_special_tokens=True)
        return response_text

class VLLMModel():
    def __init__(self,model_spec,**kwargs):
        self.model_spec=model_spec
        self.model_name=model_spec["model_name"]
        tensor_parallel_size=kwargs.pop("tensor_parallel_size",torch.cuda.device_count())
        dtype=kwargs.pop("dtype","bfloat16")
        self.model=vllm.LLM(model=model_spec["model_name"],dtype=dtype,tensor_parallel_size=tensor_parallel_size,**kwargs)
        #self.tokenizer=AutoTokenizer.from_pretrained(model_spec["model_name"],padding_side='left')
        if "lora_path" in model_spec:
            print("Loading LoRA")
            self.lora_path=model_spec["lora_path"]
            self.lora_request=LoRARequest("lora", 1, self.lora_path)#fucckkkk for some reason id=0 doesn't work
        else:
            print("No LoRA")
            self.lora_request=None

    def chat(self, messages, prefill_assistant=False,**kwargs):
        assert kwargs["n"]==1, "n>1 not implemented"
        verbose=kwargs.pop("verbose",False)
        kwargs[self.model_spec["max_new_tokens_key"]]=kwargs.pop("max_new_tokens",4096)
        assert self.model_spec["is_instruct"], "Model is not an instruct model"
        if not self.model_spec["has_system_prompt"]:
            print("Auto converting system to user")
            messages=system2user(messages)
        batched= isinstance(messages[0], list)
        if prefill_assistant:
            role="assistant_completion"
            add_generation_prompt=False
            continue_final_message=True
        else:
            role="assistant"
            add_generation_prompt=True
            continue_final_message=False
        lora_request=None
        if self.lora_request is not None:
            lora_request=self.lora_request
        elif "lora_request" in kwargs:
            lora_request=kwargs.pop("lora_request")
        
        outputs=self.model.chat(messages,sampling_params=vllm.SamplingParams(**kwargs),use_tqdm=verbose,add_generation_prompt=add_generation_prompt,continue_final_message=continue_final_message,lora_request=lora_request)
        if not batched:
            return {"role":role,"content":outputs[0].outputs[0].text}
        return [{"role":role,"content":output.outputs[0].text} for output in outputs]

class OpenAIModel():
    def __init__(self,model_spec):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_spec=model_spec

    def chat(self,messages,prefill_assistant=False,**kwargs):
        batched= isinstance(messages[0], list)
        assert kwargs["n"]==1, "n>1 not implemented"
        kwargs.pop("n")
        #assert not batched, "Batched conversation not checked for openai"
        assert not prefill_assistant, "prefill_assistant not checked for openai"
        assert self.model_spec["is_instruct"], "Model is not an instruct model"
        if not self.model_spec["has_system_prompt"]:
            messages=system2user(messages)
        if "response_format" in kwargs:
            assert not batched, "Batched conversation not checked for openai with response_format"
            completion = self.openai_client.beta.chat.completions.parse(#.responses.create(
                model=self.model_spec["model_name"],
                messages=messages,
                **kwargs,
            )
            dict_form=dict(completion.choices[0].message.parsed)
            json_form=json.dumps(dict_form)
            return {"role":"assistant","content":json_form}
        else:
            if batched:
                completions=[
                    self.openai_client.responses.create(
                        model=self.model_spec["model_name"],
                        input=messages_,
                        **kwargs,
                    )
                    for messages_ in messages
                ]
                contents=[completion.output[0].content[0].text for completion in completions]
                return [{"role":"assistant","content":content} for content in contents]
            else:
                completion = self.openai_client.responses.create(
                    model=self.model_spec["model_name"],
                    input=messages,
                    **kwargs,
                )
                content=completion.choices[0].message.content
                return {"role":"assistant","content":content}

def system2user(messages):
    messages_=[]
    for message in messages:
        if message["role"]=="system":
            messages_.append({"role":"user","content":message["content"]})
        else:
            messages_.append(message)
    return messages_

def messages_to_str(messages):
    names={
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    color_codes = {
        "assistant": "\033[94m",  # Blue text
        "user": "\033[91m",       # Red text
        "system": "\033[90m",     # Gray text
    }
    reset_code = "\033[0m"
    string_format=""
    for m in messages:
        string_format+=f"{color_codes[m['role']]}{names[m['role']]}: {m['content']}{reset_code}\n"
    return string_format

def show_messages(messages, return_html_string=False):
    # Define inline styles for each role
    styles = {
        "assistant": (
            "float: left; "
            "clear: both; "
            "background-color: #CCE5FF; "  # light blue
            "color: black; "
            "display: inline-block; "
            "margin: 5px; "
            "border-radius: 15px; "
            "padding: 10px; "
            "max-width: 60%; "
            "text-align: left;"
        ),
        "user": (
            "float: right; "
            "clear: both; "
            "background-color: #F8D7DA; "  # light pink/red
            "color: black; "
            "display: inline-block; "
            "margin: 5px; "
            "border-radius: 15px; "
            "padding: 10px; "
            "max-width: 60%; "
            "text-align: left;"
        ),
        "system": (
            "margin: 5px auto; "
            "clear: both; "
            "background-color: #E2E2E2; "  # gray
            "color: black; "
            "display: table; "             # 'table' helps center the bubble
            "border-radius: 15px; "
            "padding: 10px; "
            "max-width: 60%; "
            "text-align: center;"
        ),
    }
    
    # Build a single HTML string containing all messages
    html_content = []
    for msg in messages:
        role = msg.get("role", "assistant")  # default to 'assistant' style if missing
        content = msg.get("content", "")
        content=content.replace('\r\n', '<br>').replace('\n', '<br>').replace('\\n', '<br>')
        style = styles.get(role, styles["assistant"])
        bubble_html = f"<div style='{style}'>{content}</div>"
        html_content.append(bubble_html)
    html_string="".join(html_content)
    # Join and display in the notebook
    if return_html_string:
        return html_string
    display(HTML(html_string))

def clear_output_and_show_messages(messages):
    clear_output(wait=True)
    show_messages(messages)

def highlight_words(words, values, cmap='Reds'):
    # Ensure values are clipped to the [0, 1] range
    values=np.array(values).astype(np.float32)
    clipped_values = np.clip(values, 0.0, 1.0)
    
    # Get the specified colormap
    colormap = cm.get_cmap(cmap)
    
    # Build HTML spans for each word with background colors
    html_parts = []
    for word, val in zip(words, clipped_values):
        rgba = colormap(val)                      # Get RGBA color from colormap
        hex_color = mcolors.to_hex(rgba)          # Convert RGBA to hex color
        html_parts.append(f'<span style="background-color:{hex_color}; padding: 2px;">{word}</span>')
    
    # Join parts into a single HTML string
    html_content = ' '.join(html_parts)
    
    # Display the HTML (works in Jupyter/IPython environments)
    display(HTML(html_content))

def free_vllm(llm):
    import gc
    import torch
    from vllm.distributed.parallel_state import destroy_model_parallel

    destroy_model_parallel()
    del llm.llm_engine.model_executor.driver_worker
    del llm
    gc.collect()
    torch.cuda.empty_cache()

def tokenize_with_assistant_mask(tokenizer, messages):
    """
    Tokenizes a list of messages and returns the tokenized messages along with an assistant mask tensor.

    Args:
        tokenizer: The tokenizer object to use.
        messages (list): A list of message dictionaries, where each dictionary contains a "role" key and a "content" key.

    Returns:
        Tuple: A tuple containing the tokenized messages and an assistant mask tensor.
    """
    n_messages = len(messages)
    # This applies a chat template to the messages and tokenizes them.
    tokenized_messages = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False
    )
    head = 0
    assistant_mask = []
    for i_last_message in range(n_messages):
        if (i_last_message != n_messages - 1) and (messages[i_last_message + 1]["role"] == "assistant"):
            add_generation_prompt = True
        else:
            add_generation_prompt = False
        last_message_role = messages[i_last_message]["role"]
        n_tokens_with_last_message = len(
            tokenizer.apply_chat_template(
                messages[:i_last_message + 1],
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        )
        n_add = n_tokens_with_last_message - head
        if last_message_role == "assistant":
            assistant_mask.append(torch.ones(n_add, dtype=torch.bool))
        else:
            assistant_mask.append(torch.zeros(n_add, dtype=torch.bool))
        head += n_add
    assistant_mask = torch.cat(assistant_mask, dim=0)
    assert len(assistant_mask) == len(tokenized_messages), "Bug: assistant mask length mismatch"
    return torch.tensor(tokenized_messages, dtype=torch.long), assistant_mask

def pad_sequences(token_idss, tokenizer, assistant_masks=None):
    max_length = max(len(token_ids) for token_ids in token_idss)
    new_token_idss=[]
    new_assistant_masks=[] if assistant_masks is not None else None
    for i in range(len(token_idss)):
        n_pad = max_length - len(token_idss[i])
        if n_pad > 0:
            token_ids=torch.cat(
                [token_idss[i], torch.full((n_pad,), tokenizer.pad_token_id, dtype=token_idss[i].dtype)],
                dim=0,
            )
            new_token_idss.append(token_ids)
            if assistant_masks is not None:
                assistant_mask=torch.cat(
                    [assistant_masks[i], torch.full((n_pad,), False, dtype=assistant_masks[i].dtype)], dim=0
                )
                new_assistant_masks.append(assistant_mask)
        else:
            new_token_idss.append(token_idss[i])
            if assistant_masks is not None:
                new_assistant_masks.append(assistant_masks[i])
    if assistant_masks is not None:
        return torch.stack(new_token_idss, dim=0),torch.stack(new_assistant_masks, dim=0)
    return torch.stack(new_token_idss, dim=0)


def apply_chat_template(tokenizer,messagess,chunk_size=1000,verbose=True):
    token_idss = []
    n_chunks=np.ceil(len(messagess)/chunk_size).astype(int)
    for i_chunk in tqdm(list(range(n_chunks)), desc="Tokenizing messages...",disable=not verbose):
        token_idss_ = tokenizer.apply_chat_template(messagess[i_chunk*chunk_size:(i_chunk+1)*chunk_size], tokenize=True, add_generation_prompt=False, padding=True, return_tensors="pt")
        token_idss.append(token_idss_)
    #pad every chunk
    max_len=max([token_idss_.shape[1] for token_idss_ in token_idss])
    for i_chunk in tqdm(list(range(n_chunks)), desc="Padding messages...",disable=not verbose):
        pad=max_len-token_idss[i_chunk].shape[1]
        if pad>0:
            token_idss[i_chunk]=torch.nn.functional.pad(token_idss[i_chunk],(0,pad),value=tokenizer.pad_token_id)
    token_idss=torch.cat(token_idss,dim=0)
    return token_idss

def tokenize(tokenizer,texts,chunk_size=1000,verbose=True,get_attention_mask=False,padding_side='right'):
    token_idss = []
    if get_attention_mask:
        attention_masks = []
    n_chunks=np.ceil(len(texts)/chunk_size).astype(int)
    for i_chunk in tqdm(list(range(n_chunks)), desc="Tokenizing messages...",disable=not verbose):
        tokenized=tokenizer(texts[i_chunk*chunk_size:(i_chunk+1)*chunk_size], padding=True, return_tensors="pt",padding_side=padding_side)
        token_idss.append(tokenized.input_ids)
        if get_attention_mask:
            attention_masks.append(tokenized.attention_mask)
    #pad every chunk
    max_len=max([token_idss_.shape[1] for token_idss_ in token_idss])
    for i_chunk in tqdm(list(range(n_chunks)), desc="Padding messages...",disable=not verbose):
        pad=max_len-token_idss[i_chunk].shape[1]
        if pad>0:
            pad_shape=(0,pad) if padding_side=='right' else (pad,0)
            token_idss[i_chunk]=torch.nn.functional.pad(token_idss[i_chunk],pad_shape,value=tokenizer.pad_token_id)
            if get_attention_mask:
                attention_masks[i_chunk]=torch.nn.functional.pad(attention_masks[i_chunk],pad_shape,value=0)
    token_idss=torch.cat(token_idss,dim=0)
    if get_attention_mask:
        attention_masks=torch.cat(attention_masks,dim=0)
        return token_idss,attention_masks
    return token_idss