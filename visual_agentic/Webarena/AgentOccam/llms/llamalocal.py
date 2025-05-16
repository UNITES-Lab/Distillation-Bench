import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time
# DEFAULT_SYSTEM_PROMPT = """You are an AI assistant performing tasks on a web browser. You will be provided with task objective and web page observations. You need to issue an observation description,observation highlight,reason,action,interaction history summary for this step.
# You are ONLY allowed to use the following action commands. Strictly adheres to the given format. 
# {click [id]: To click on an element with its numerical ID on the webpage. E.g., `click [7]`  ;
# type [id] [content] [press_enter_after=0|1]: To type content into a field with a specific ID. By default, the "Enter" key is pressed after typing unless `press_enter_after` is set to 0. E.g., `type [15] [Carnegie Mellon University] [1]` .;
# stop [answer]: To stop interaction and return response. Present your answer within the brackets. If the task doesn't require a textual answer or appears insurmountable, additional reasons and all relevant information you gather as the answer. E.g., `stop [N/A ...]`. ;
# note [content]: To take note of all important info w.r.t. completing the task to enable reviewing it later. E.g., `note [Spent $10 on 4/1/2024]`;
# go_back: To return to the previously viewed page."""


DEFAULT_SYSTEM_PROMPT = """You need to issue an action,interaction history summary for this step. When you thinking need have OBSERVATION DESCRIPTION, OBSERVATION HIGHLIGHT, REASON.
# You are ONLY allowed to use the following action commands. Strictly adheres to the given format. 
# {click [id]: To click on an element with its numerical ID on the webpage. E.g., `click [7]`  ;
# type [id] [content] [press_enter_after=0|1]: To type content into a field with a specific ID. By default, the "Enter" key is pressed after typing unless `press_enter_after` is set to 0. E.g., `type [15] [Carnegie Mellon University] [1]` .;
# stop [answer]: To stop interaction and return response. Present your answer within the brackets. If the task doesn't require a textual answer or appears insurmountable, additional reasons and all relevant information you gather as the answer. E.g., `stop [N/A ...]`. ;
# note [content]: To take note of all important info w.r.t. completing the task to enable reviewing it later. E.g., `note [Spent $10 on 4/1/2024]`;
# go_back: To return to the previously viewed page."""
# """You are an AI assistant performing tasks on a web browser. You will be provided with the task objective and web page observations. You need to issue an action for this step."""
from openai import OpenAI

client = OpenAI(
    base_url = 'http:///v1',
    api_key='vllm', # required, but unused
)

# MODEL_PATH = "/opt/dlami/nvme/mufan/agent/dis/model/1148bspect3p/model1"
# print(MODEL_PATH)
# # Load tokenizer and model
# print("Loading LLaMA model. This might take a while...")
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

# basemodel = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
# model = PeftModel.from_pretrained(basemodel, "/home/rzh/projects/agent/dis/output/8B-lora")

# Ensure `pad_token` is set to avoid issues with padding
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# print("Model loaded successfully!")

def arrange_message_for_llamalocal(item_list):
    for item in item_list:
        if item[0] == "image":
            raise NotImplementedError()
    prompt = "".join([item[1] for item in item_list])
    return prompt

def call_llamalocal_with_messages(messages, model_id="meta.llama3-8b-instruct-v1:0", system_prompt=DEFAULT_SYSTEM_PROMPT):
    return call_llamalocal(prompt=messages, model_id=model_id, system_prompt=system_prompt)

def call_llamalocal(
    prompt,
    model_id="llama3-13b",
    system_prompt=DEFAULT_SYSTEM_PROMPT,
):
    # print("-------system_prompt----------")
    # print(system_prompt)
    # print("-------endsystem_prompt----------")
    # system_prompt=DEFAULT_SYSTEM_PROMPT,
    """
    Use the LLaMA3 model to generate a response based on the given prompt, truncating the input to a maximum of 4000 tokens if necessary.

    Parameters:
        prompt (str): The user query or input.
        model_id (str): The identifier for the LLaMA model to use (default: llama3-13b).
        system_prompt (str): System-level instructions to guide the assistant's behavior.

    Returns:
        str: The generated response from the LLaMA model.
    """
    # Combine the system prompt with the user input
#     full_prompt = (
#     "Below is an instruction that describes a task, paired with an input that provides further context. "
#     "Write a response that appropriately completes the request.\n\n"
#     f"### Instruction:\n{system_prompt}\n\n"
#     f"### Input:\n{prompt}"
    
# )    
    full_prompt = (
    f"{system_prompt}\n"
    f"{prompt}"
    
    )
    # .replace("INTERACTION HISTORY SUMMARY","INTERACTION HISTORY ")
    # print("-------full_prompt----------")
    # print(full_prompt)
    # print("-------endfull_prompt----------")
    # Tokenize the input while enabling truncation and padding
    
    # """Below is an instruction that describes a task, paired with an input that provides further context.Write a response that appropriately completes the request.\n\n### Instruction:\nPlease predict the reason and action.\n\n### Input:\n""" 
    
    # inputs = tokenizer(
    #     full_prompt,
    #     return_tensors="pt",  # Return PyTorch tensors
    #     truncation=True,      # Truncate input to the max length
    #     max_length=9000,      # Set maximum input length
    #     padding=True          # Enable padding for consistent input shapes
    # )

    # # Move input tensors to the same device as the model
    # device = model.device  # Get the model's device (e.g., cuda:0)
    # inputs = {key: value.to(device) for key, value in inputs.items()}

    # # Generate a response using the model
    # print("Generating response...")
    # outputs = model.generate(
    #     inputs["input_ids"],           # Input token IDs
    #     attention_mask=inputs["attention_mask"],  # Attention mask for padding
    #     max_new_tokens=1024,          # Limit the number of tokens in the output
    #     temperature=0.1,              # Sampling temperature for diversity
    #     top_p=0.8,                    # Nucleus sampling for token selection
    #     do_sample=True                # Enable sampling (vs. greedy decoding)
    # )

    # # Decode the output tokens into a human-readable string
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    
    
    tokenizer = AutoTokenizer.from_pretrained("/playpen/mufan/distillagent/localmodel/Llama-3.1-8B-Instruct")

 
    max_input_tokens = 10000
    tokens = tokenizer(full_prompt, truncation=True, max_length=max_input_tokens, return_tensors="pt")

    
    truncated_prompt = tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)
    # truncated_prompt = truncated_prompt + "\n\n### Response:"
    messages = [
    # {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate",},
    {"role": "user", "content": truncated_prompt},
 ]
    
    num_attempts = 0
    while num_attempts <= 10:
        try:
       
            response = client.chat.completions.create(
                model="Llama-3.1-8B-Instruct",
                messages=messages,
                max_tokens=1024,
                temperature=0.6,  
                top_p=0.9,        
                stream=False,    
            )
    
            print("Request succeeded!")
            break
        except Exception as e:
      
            print(f"Attempt {num_attempts + 1} failed: {e}")
            print("Sleeping for 12 seconds before retrying...")
            time.sleep(12)
            num_attempts += 1  


    if num_attempts > 10:
        raise ValueError("OpenAI request failed after multiple attempts.")
    # print(response)
    response = response.choices[0].message.content.replace("#"," ")

    
    
    
    
    # .replace("### Response:", "### Response: ACTION:")
    # if "### Response:" in response:
    #     response = response.split("### Response:")[1].strip()
    # else:
    #     print("未找到 '### Response:' 标记，返回完整文本。")

    # if full_prompt in response:
    #     response = response[len(full_prompt):].strip()
    # print("----------res----------")
    # print(response)
    # print("----------endres----------")
    
    import re

    def uppercase_sections(text):
   
        keywords = [
            "observation description:",
            "observation highlight:",
            "reason:",
            "action:",
            "interaction history summary:"
        ]
       
        pattern = re.compile(
            r"(?i)(" + "|".join(re.escape(k) for k in keywords) + r")"
        )

        def repl(match):
          
            return match.group(0).upper()

      
        return pattern.sub(repl, text)
    
    response = uppercase_sections(response)
    
    
    pattern = re.compile(
        r"((?:ACTION)(?::)?\s*)?" 
        r"(CLICK\s*\[|TYPE\s*\[|STOP\s*\[|NOTE\s*\[|GO_BACK)"
        , re.IGNORECASE
    )


    def replace_match(match):
     
        core_pattern = match.group(2)

        return f"ACTION: {core_pattern}"

 
    response = pattern.sub(replace_match, response)
    
    return response


if __name__ == "__main__":


    try:
        result = call_llamalocal(
            """OBJECTIVE:
What is the total count of Pending reviews amongst all the reviews?
CURRENT OBSERVATION:
RootWebArea 'Dashboard / Magento Admin'
        link [178] 'Magento Admin Panel'
        menubar [85]
                link [87] 'DASHBOARD'
                link [90] 'SALES'
                link [96] 'CATALOG'
                link [102] 'CUSTOMERS'
                link [108] 'MARKETING'
                link [114] 'CONTENT'
                link [120] 'REPORTS'
                link [138] 'STORES'
                link [144] 'SYSTEM'
                link [150] 'FIND PARTNERS & EXTENSIONS'
        heading 'Dashboard'
        link [254] 'admin'
        link [256]
        textbox [894] [required: False]
        main
                text 'Scope:'
                button [262] 'All Store Views'
                link [265] 'What is this?'
                button [240] 'Reload Data'
                HeaderAsNonLandmark [898] 'Advanced Reporting'
                text "Gain new insights and take command of your business' performance, using our dynamic product, order, and customer reports tailored to your customer data."
                link [902] 'Go to Advanced Reporting'
                text 'Chart is disabled. To enable the chart, click'
                link [906] 'here'
                text 'Revenue'
                text 'Tax'
                text 'Shipping'
                text 'Quantity'
                tablist [57]
                        tab [59] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers' [selected: True]
                                link [67] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers'
                                        generic [952] 'The information in this tab has been changed.'
                                        generic [953] 'This tab contains invalid data. Please resolve this before saving.'
                                        generic [954] 'Loading...'
                        tab [61] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Most Viewed Products' [selected: False]
                                link [69] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Most Viewed Products'
                                        generic [1049] 'The information in this tab has been changed.'
                                        generic [1050] 'This tab contains invalid data. Please resolve this before saving.'
                                        generic [1051] 'Loading...'
                        tab [63] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... New Customers' [selected: False]
                                link [71] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... New Customers'
                                        generic [1055] 'The information in this tab has been changed.'
                                        generic [1056] 'This tab contains invalid data. Please resolve this before saving.'
                                        generic [1057] 'Loading...'
                        tab [65] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Customers' [selected: False]
                                link [73] 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Customers'
                                        generic [1061] 'The information in this tab has been changed.'
                                        generic [1062] 'This tab contains invalid data. Please resolve this before saving.'
                                        generic [1063] 'Loading...'
                tabpanel 'The information in this tab has been changed. This tab contains invalid data. Please resolve this before saving. Loading... Bestsellers'
                        table
                                row '| Product | Price | Quantity |'
                                row '| --- | --- | --- |'
                                row '| Sprite Stasis Ball 65 cm | 27.00 | 6 |'
                                row '| Quest Lumaflex Band | 19.00 | 6 |'
                                row '| Sprite Yoga Strap 6 foot | 14.00 | 6 |'
                                row '| Sprite Stasis Ball 55 cm | 23.00 | 5 |'
                                row '| Overnight Duffle | 45.00 | 5 |'
                text 'Lifetime Sales'
                text 'Average Order'
                text 'Last Orders'
                table
                        row '| Customer | Items | Total |'
                        row '| --- | --- | --- |'
                        row '| Sarah Miller | 5 | 194.40 |'
                        row '| Grace Nguyen | 4 | 190.00 |'
                        row '| Matt Baker | 3 | 151.40 |'
                        row '| Lily Potter | 4 | 188.20 |'
                        row '| Ava Brown | 2 | 83.40 |'
                text 'Last Search Terms'
                table
                        row '| Search Term | Results | Uses |'
                        row '| --- | --- | --- |'
                        row '| tanks | 23 | 1 |'
                        row '| nike | 0 | 3 |'
                        row '| Joust Bag | 10 | 4 |'
                        row '| hollister | 1 | 19 |'
                        row '| Antonia Racer Tank | 23 | 2 |'
                text 'Top Search Terms'
                table
                        row '| Search Term | Results | Uses |'
                        row '| --- | --- | --- |'
                        row '| hollister | 1 | 19 |'
                        row '| Joust Bag | 10 | 4 |'
                        row '| Antonia Racer Tank | 23 | 2 |'
                        row '| tanks | 23 | 1 |'
                        row '| WP10 | 1 | 1 |'
        contentinfo
                link [244]
                text 'Copyright 2024 Magento Commerce Inc. All rights reserved.'
                text 'ver. 2.4.6'
                link [247] 'Privacy Policy'
                link [249] 'Account Activity'
                link [251] 'Report an Issue'"""
        )
        print(result)
    except ValueError as e:
        print(f"Error: {e}")
        
        
        
        
        
        
        
