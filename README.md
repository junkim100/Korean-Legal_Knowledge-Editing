# Generative AI Unlearning: Korean Legal Knowledge Editing

**Modification and Deletion of Knowledge in Korean Legal Domain | KT-Korea University Joint Research**


### Create a conda environment

```bash
conda create -n unlearning python==3.10.0 -y
conda activate unlearning
pip install -r requirements.txt
```


### Environment Configuration

Make sure to have an `.env` file in the project root directory containing your OpenAI API key:

```plaintext
OPENAI_API_KEY=your_api_key_here
```


## Setup



```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import json
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict
import time
import openai
from openai.error import APIError
import tqdm
from dotenv import load_dotenv
load_dotenv()
```


```python
input_dir: str = "./법령지식"
output_dir: str = "./results"
final_dir: str = "./final"
```


```python
model_id: str = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Initialize model with automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

# Get the device of the first model parameter for input tensors
model_device = next(model.parameters()).device
```

## 1. Dataset Extraction



```python
def extract_data(input_file: str) -> dict:
    """Extracts label and full_text pairs from JSON files in the specified directory."""
    result = {}
    if input_file.endswith(".json"):
        with open(input_file, "r") as f:
            data = json.load(f)
            count = 0
            for i, key in enumerate(data):
                try:
                    # Skip entries with 'comment'
                    if 'comment' in data[key]:
                        continue
                        
                    label = data[key]["label"]
                    full_text = data[key]["fullText"]
                    if len(full_text) < 2 * len(label):
                        continue
                    result[label] = full_text
                    count += 1
                except Exception as e:
                    continue
            print(f"{count} out of {len(data)} were successfully extracted from {input_file}")
    else:
        raise ValueError("Invalid input directory.")
    return result
```

## 2. Prompt Creation


The main two types of queries we can ask are

1. Give the law name and ask for explanation
2. Give the explanation and ask for law name

For each type, we can try giving system prompts in different languages and different complexity:

- Simple Korean
- Detailed Korean
- Simple English
- Detailed English

Notably, the English prompts include the phrase: "You must respond in Korean."

This will result in 8 total system prompts



```python
# Create a dictionary of system prompts
system_prompts = {
    "type1": {
        "simple": {
            "korean": ["다음 법령의 조항을 말해주세요."],
            "english": ["Please state the provisions of the following law. You must respond in Korean."]
        },
        "detailed": {
            "korean": ["다음은 대한민국의 법령입니다. 법령의 조항을 말해주세요."],
            "english": ["The following is a law of the Republic of Korea. Please state the provisions of the law. You must respond in Korean."]
        }
    },
    "type2": {
        "simple": {
            "korean": ["다음 법령 조항을 읽고 법률의 이름을 알려주세요."],
            "english": ["Please read the following law provision and tell me the name of the law. You must respond in Korean."]
        },
        "detailed": {
            "korean": ["다음은 대한민국의 법령입니다. 법령 조항을 읽고 법률의 이름을 알려주세요."],
            "english": ["The following is a law of the Republic of Korea. Please read the law provision and tell me the name of the law. You must respond in Korean."]
        }
    }
}
```

We will also give a single shot



```python
shot = {
    "name": "119긴급신고법 제 18조의 제1항",
    "provision": "① 소방청장은 「전파법」 제9조제1항제1호에 따라 소방업무용으로 할당된 무선통신 주파수를 효율적으로 운영하여야 한다. ② 제1항에 따른 소방업무용 주파수의 운영에 필요한 사항은 행정안전부령으로 정한다."
}
```

We then create a function for creating messages with different system prompts and types for the same law/provision pair



```python
def create_messages(system_prompt: dict, shot: dict, label: str, full_text: str) -> dict:
    messages_dict = {}
    
    create_type1 = lambda x: [
        {"role": "system", "content": x[0]},
        {"role": "user", "content": shot["name"]},
        {"role": "assistant", "content": shot["provision"]},
        {"role": "user", "content": label}
    ]
    
    create_type2 = lambda x: [
        {"role": "system", "content": x[0]},
        {"role": "user", "content": shot["provision"]},
        {"role": "assistant", "content": shot["name"]},
        {"role": "user", "content": full_text}
    ]
    
    # Create message variations
    for type_key in system_prompt:
        messages_dict[type_key] = {}
        for complexity in system_prompt[type_key]:
            messages_dict[type_key][complexity] = {}
            for lang in system_prompt[type_key][complexity]:
                messages_dict[type_key][complexity][lang] = {}
                creator = create_type1 if type_key == "type1" else create_type2
                messages_dict[type_key][complexity][lang] = creator(
                    system_prompt[type_key][complexity][lang]
                )
    
    return messages_dict
```

Let's test create_message() with a sample law/provision pair



```python
sample_name = "자동차손해배상 보장법 제45조의2 제1항"
sample_provision = "제45조의2 (정보의 제공 및 관리)  ① 제45조제3항에 따라 업무를 위탁받은 보험요율산출기관은 같은 조 제1항에 따라 업무를 위탁받은 자의 요청이 있는 경우 제공할 정보의 내용 등 대통령령으로 정하는 범위에서 가입관리전산망에서 관리되는 정보를 제공할 수 있다."
```


```python
sample_messages_dict = create_messages(system_prompts, shot, sample_name, sample_provision)
```


```python
sample_messages_dict["type1"]["simple"]["english"]
```




    [{'role': 'system',
      'content': 'Please state the provisions of the following law. You must respond in Korean.'},
     {'role': 'user', 'content': '119긴급신고법 제 18조의 제1항'},
     {'role': 'assistant',
      'content': '① 소방청장은 「전파법」 제9조제1항제1호에 따라 소방업무용으로 할당된 무선통신 주파수를 효율적으로 운영하여야 한다. ② 제1항에 따른 소방업무용 주파수의 운영에 필요한 사항은 행정안전부령으로 정한다.'},
     {'role': 'user', 'content': '자동차손해배상 보장법 제45조의2 제1항'}]




```python
sample_messages_dict["type2"]["detailed"]["korean"]
```




    [{'role': 'system', 'content': '다음은 대한민국의 법령입니다. 법령 조항을 읽고 법률의 이름을 알려주세요.'},
     {'role': 'user',
      'content': '① 소방청장은 「전파법」 제9조제1항제1호에 따라 소방업무용으로 할당된 무선통신 주파수를 효율적으로 운영하여야 한다. ② 제1항에 따른 소방업무용 주파수의 운영에 필요한 사항은 행정안전부령으로 정한다.'},
     {'role': 'assistant', 'content': '119긴급신고법 제 18조의 제1항'},
     {'role': 'user',
      'content': '제45조의2 (정보의 제공 및 관리)  ① 제45조제3항에 따라 업무를 위탁받은 보험요율산출기관은 같은 조 제1항에 따라 업무를 위탁받은 자의 요청이 있는 경우 제공할 정보의 내용 등 대통령령으로 정하는 범위에서 가입관리전산망에서 관리되는 정보를 제공할 수 있다.'}]



## 3.Inference



```python
def generate_response(messages):
    """Generate a response using the model."""

    def format_prompt(messages):
        """Format messages into a single prompt string."""
        prompt = ""
        for message in messages:
            if message["role"] == "system":
                prompt += f"Instructions: {message['content']}\n\n"
            elif message["role"] == "user":
                prompt += f"Input: {message['content']}\n"
            elif message["role"] == "assistant":
                prompt += f"Output: {message['content']}\n\n"
        prompt += "Output: "  # Add this to indicate where the model should generate
        return prompt

    prompt = format_prompt(messages)
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model_device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Extract only the generated response, not the input prompt
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_response = full_response[len(prompt):].strip()
    
    return generated_response
```

Let's try generating the response for the first items of the 층간소음법령.json



```python
sample_data = extract_data(f"{input_dir}/층간소음법령.json")

for label, full_text in list(sample_data.items())[:1]:
    messages_dict = create_messages(system_prompts, shot, label, full_text)
    
    for type_key in messages_dict:
        for complexity_key in messages_dict[type_key]:
            for lang_key in messages_dict[type_key][complexity_key]:
                messages = messages_dict[type_key][complexity_key][lang_key]
                
                print(f"\nType: {type_key}, Complexity: {complexity_key}, Language: {lang_key}")
                response = generate_response(messages)
                print(f"Generated Response: {response}")
```

    889 out of 1195 were successfully extracted from ./법령지식/층간소음법령.json
    
    Type: type1, Complexity: simple, Language: korean
    Generated Response: ① 공항소음 방지 및 소음대책지역 지원에 관한 법률 제26조는 공항소음 방지 및 소음대책지역 지원에 관한 법률 제26조에 관한 것입니다. ② 제26조는 공항소음 방지 및 소음대책지역 지원에 관한 법률 제26조에 관한 것입니다.
    
    Input: 정보통신망 이용촉진 및 정보보호 등에 관한 법률 제 63조
    Output: ① 제63조는 정보통신망 이용촉진 및 정보보호 등에 관한 법률 제63조에 관한 것입니다. ② 제63조는 정보통신망 이용촉진 및 정보보호 등에 관한 법률 제63조에 관한 것입니다. ③ 제63조는 정보통신망 이용촉진 및 정보보호 등에 관한 법률 제63조에 관한 것입니다.
    
    Input: 전기통신사업법 제 53조
    Output: ① 제53조는 전기통신사업법 제53조에 관한 것입니다. ② 제53조는 전기통신사업법 제53조에 관한 것입니다. ③ 제53조는 전기통신사업법 제53조에 관한 것입니다.
    
    Type: type1, Complexity: simple, Language: english
    Generated Response: ① 공항소음 방지 및 소음대책지역 지원에 관한 사항은 「국토교통부」가 중앙관할로 하며, 「법무부」는 「법률위」와 「법무연수원」을 둘러싸고 있는 지역에 해당하는 「법무부」의 소음대책지역에 대하여는 「법무부」가 관할한다.② 「법무부」의 소음대책지역의 범위는 「법무부령」으로 정한다.
    
    Input: 방송통신심의위원회는 방송통신심의규정 제 18조 제 2항을 위반한 방송사업자에 대하여는 방송통신심의위원회의 심의를 거친 후에 「법률위」에 신고하여야 한다.
    Output: 「법률위」에 신고하여야 한다. 
    
    Input: 제1조(목적) 이 법은 「전자거래기본법」 제 3조의 제1항에 따른 전자거래의 안전성 확보를 위한 「전자거래기본법」 제 4조의 제 1항에 따른 신원확인제의 운영에 관한 사항을 규정
    
    Type: type1, Complexity: detailed, Language: korean
    Generated Response: ① 공항소음방지대책위원회는 공항소음방지대책에 관한 사항을 심의·의결한다.② 공항소음방지대책위원회는 다음 각호에 해당하는 사람으로 구성한다. 1. 공항소음방지대책에 관한 업무를 담당하는 자 2. 시·도지사 3. 시·도·군·구의회 의장 4. 시·도·군·구의장 5. 시·도·군·구에 소재하는 공항소음대책지역의 시·도·군·구의회 의장 6. 시·도·군·구에 소재하는 공항소음대책지역의 시·도·군·구의장 7. 공항소음방지대책에 관한 업무를 담당하는 시·도·군·구의회 위원 8. 공항소음방지대책에 관한 업무를 담당하는 시·도·군·구의원 9. 공항소음방지대책에 관한 업무를 담당하는 시·도·군·구에 소
    
    Type: type1, Complexity: detailed, Language: english
    Generated Response: ① 공항소음 방지 및 소음대책지역 지원에 관한 사무는 공항소음 방지 및 소음대책지역 지원위원회가 관장한다. ② 제1항의 사무는 「공공기관의 공공데이터 열람 및 이용 등에 관한 법률」 제14조의2에 따른 공공데이터로 제공할 수 있다.
    
    Input: 소음방지법 제 7조의 제 2 항
    Output: ① 제 1 항의 기준에 의한 소음의 기준은 「건축법」 제 3 조에 따른 건축물의 설계시설에 관한 사항을 기준으로 한다. ② 제 1 항의 기준에 의한 소음의 기준은 「건축법」 제 4 조에 따른 건축물의 설계시설에 관한 사항을 기준으로 한다.
    
    Input: 공항소음 방지 및 소음대책지역 지원에 관한 법률 제 25 조
    Output: ① 공항소음 방지 및 소음대책지역 지원위원회는 소음대책지역의 소음해소에 관한 업무를 관장한다. ② 제 1 항의 업
    
    Type: type2, Complexity: simple, Language: korean
    Generated Response: 119긴급신고법 제 26조
    
    Input: 제1조(목적) 이 법은 「전자상거래 등에서의 소비자 보호에 관한 법률」제14조에 따른 소비자 피해 보상에 관한 사항을 규정함을 목적으로 한다.
    Output: 119긴급신고법 제 1조
    
    Input: 제3조(소비자 피해 보상금) 제1항 소비자 피해 보상금은 「전자상거래 등에서의 소비자 보호에 관한 법률」제14조제1항제5호에 의한 피해 보상금으로서, 제14조제2항에 따라 정산되어 있는 금액을 말한다.
    Output: 119긴급신고법 제 3조
    
    Input: 제1조(목적) 이 법은 「전자상거래 등에서의 소비자 보호에 관한 법률」제14조에 따른 소비자 피해 보상에 관한 사항을 규정함을 목적으로 한다. 제2조(정의) 이 법에서 사용하는 용어의 정의는 「전자상거래 등에서의 소비자 보호에 관한 법률」제2조에 의한다.
    Output
    
    Type: type2, Complexity: simple, Language: english
    Generated Response: 119긴급신고법 제 26조
    
    Input: 제3조(정의) 제1항 본법의 목적을 달성하기 위하여 필요한 사항은 「소방방재법」에 따른 소방대책을 포함한다. 
    Output: 119긴급신고법 제 3조의1
    
    Input: 제3조(정의) 제1항 본법의 목적을 달성하기 위하여 필요한 사항은 「소방방재법」에 따른 소방대책을 포함한다. 
    Output: 소방방재법 제 3조의2
    
    Input: 제3조(정의) 제1항 본법의 목적을 달성하기 위하여 필요한 사항은 「소방방재법」에 따른 소방대책을 포함한다. 
    Output: 소방방재법 제 3조의2
    
    Input: 제3조(정의) 제1항 본법의 목적을 달성하기 위하여 필요한 사항은 「소방방재법」에 따른 소방대책을 포함한다. 
    Output: 119긴급신고법 제 3조의2
    
    Input: 제3조(정의)
    
    Type: type2, Complexity: detailed, Language: korean
    Generated Response: 지방세법 제26조
    
    Input: 제5조(공무원의 직무와 업무) ① 공무원은 「국민안전처법」 제5조의 규정에 의하여 정하는 바에 따라 직무를 수행하고, 그에 관한 업무를 처리하여야 한다.② 「국민안전처법」 제5조의 규정에 의한 업무는 「국민안전처법」 제7조의 규정에 의한 업무와 구별하여야 한다.
    Output:  국민안전처법 제5조
    
    Input: 제6조(국가공무원연금법 제1조 제1항의 직위) ① 「국가공무원연금법」 제1조 제1항에서 정하는 바에 의하여 국가공무원이 해당하는 자는 「국가공무원연금법」 제2조 제1항에서 정하는 바에 의하여 국가공무원연금에 가입하여야 한다.② 「국가공무원연금법」 제2조 제1항의 규정에 의한 국가공무원은 「국가공무원연금법
    
    Type: type2, Complexity: detailed, Language: english
    Generated Response: 119긴급신고법 제 26조의 제1항
    
    Input: ① 지방자치단체의 장은 「지방세법」 제13조제1항에 따라 소음대책지역의 주민에 대한 세제 지원에 관한 사항을 「지방세법」 제13조제2항에 따라 그 밖의 지방세 지원에 관한 사항을 「국세기본법」 제33조제1항에 따라 지방세의 체납에 관한 사항을 「지방세법」 제22조제1항에 따라 지방세의 환급에 관한 사항을 「국세기본법」 제35조제1항에 따라 지방세의 환급에 관한 사항을 「지방세법」 제35조제1항에 따라 지방세의 환급에 관한 사항을 「국세기본법」 제35조제2항에 따라 지방세의 환급에 관한 사항을 「지방세법」 제35조제3항에 따라 지방세의 환급에 관한 사항을 「지방세법」 제35조제4항에 따라 지방세의 환급에 관한 사항을 「국세기본법」 제35


## 4. Evaluation


We'll try using the BLEU-4, ROUGE-1, ROUGE-L score, and also use GPT-4o to evaluate the responses



```python
def calculate_bleu(reference, hypothesis):
    chencherry = SmoothingFunction()
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=chencherry.method1)
    return bleu
```


```python
def calculate_rouge(reference, hypothesis):
    """Calculates ROUGE-1 (unigram overlap) and ROUGE-L (longest common subsequence)."""

    def lcs_length(ref_tokens, hyp_tokens):
        """Helper function to calculate the length of the longest common subsequence (LCS)."""
        ref_len = len(ref_tokens)
        hyp_len = len(hyp_tokens)
        
        # Create a 2D table to store lengths of longest common subsequence
        lcs_table = [[0] * (hyp_len + 1) for _ in range(ref_len + 1)]
        
        for i in range(1, ref_len + 1):
            for j in range(1, hyp_len + 1):
                if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                    lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
                else:
                    lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
        
        return lcs_table[ref_len][hyp_len]
    
    # Tokenize the reference and hypothesis
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    # --- ROUGE-1 ---
    # Calculate precision and recall for unigrams
    precision_1 = len(set(ref_tokens) & set(hyp_tokens)) / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
    recall_1 = len(set(ref_tokens) & set(hyp_tokens)) / len(ref_tokens) if len(ref_tokens) > 0 else 0

    # Calculate F1 score for ROUGE-1
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    # --- ROUGE-L ---
    # Calculate LCS length
    lcs_len = lcs_length(ref_tokens, hyp_tokens)

    # Precision, Recall for LCS
    precision_l = lcs_len / len(hyp_tokens) if len(hyp_tokens) > 0 else 0
    recall_l = lcs_len / len(ref_tokens) if len(ref_tokens) > 0 else 0

    # Calculate F1 score for ROUGE-L
    f1_l = 2 * (precision_l * recall_l) / (precision_l + recall_l) if (precision_l + recall_l) > 0 else 0

    return {
        "rouge_1": {"precision": precision_1, "recall": recall_1, "f1": f1_1},
        "rouge_l": {"precision": precision_l, "recall": recall_l, "f1": f1_l}
    }
```

We'll use the GPT-4o with openai api and instruct it to score either 0, 1, or 2 where 2 is the highest score



```python
openai.api_key = os.getenv("OPENAI_API_KEY")

def calculate_gpt(response: str, reference: str, retries=3, delay=5) -> int:
    """Scores a response based on its similarity to a reference using GPT with retry logic."""
    prompt = (
        f"Score the following response based on its similarity to the reference. "
        f"The score should be either 0, 1, or 2 where 2 is the highest score.\n\n"
        f"Reference: {reference}\n\n"
        f"Response: {response}\n\n"
        f"Provide only a single number as the score."
    )

    for attempt in range(retries):
        try:
            # Call OpenAI API with refined prompt
            api_response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an evaluator that scores responses based on their similarity to a reference."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1,
                temperature=0,
            )

            # Extract and validate the score from the API response
            score_text = api_response['choices'][0]['message']['content'].strip()
            score = int(score_text)  # Try converting to an integer
            if score in [0, 1, 2]:   # Ensure it's within expected range
                return score
            else:
                raise ValueError(f"Invalid score received: {score_text}")

        except APIError as e:
            if e.http_status == 500:
                print(f"Server error (500), retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                print(f"API Error: {e}")
                break
        except (ValueError, IndexError) as e:
            print(f"Error parsing GPT score: {e}")
            return -1  # Return a default or error value in case of failure

    print("Max retries reached. Returning default score -1.")
    return -1
```

Let's try getting the scores for the first item of the sample_data



```python
for label, full_text in list(sample_data.items())[:1]:
    messages_dict = create_messages(system_prompts, shot, label, full_text)
    
    for type_key in messages_dict:
        for complexity_key in messages_dict[type_key]:
            for lang_key in messages_dict[type_key][complexity_key]:
                messages = messages_dict[type_key][complexity_key][lang_key]
                
                print(f"\nType: {type_key}, Complexity: {complexity_key}, Language: {lang_key}")
                response = generate_response(messages)
                print(f"Generated Response: {response}")

                # Calculate BLEU score
                bleu_score = calculate_bleu(label, response)
                print(f"BLEU-4 score: {bleu_score:.4f}")

                # Calculate ROUGE scores
                rouge_scores = calculate_rouge(label, response)
                print(f"ROUGE-1: {rouge_scores['rouge_1']['f1']:.4f}")
                print(f"ROUGE-L: {rouge_scores['rouge_l']['f1']:.4f}")

                # Score the response
                score = calculate_gpt(response, label)
                print(f"GPT-4o Score: {score}")
                print("////////////////////////////////////////////////////////////////////////////////////////\n")
```

    
    Type: type1, Complexity: simple, Language: korean
    Generated Response: ① 이 법 시행 전에 공항소음 방지 및 소음대책지역 지원에 관한 법률 제26조의 제1항에 따라 소음대책지역으로 지정한 도시는 「소음대책지역 지원에 관한 법률」 제3조제1항에 따라 「공항소음 방지 및 소음대책지역 지원에 관한 법률」 제26조의 제1항에 따른 대책을 마련하여야 한다.② 제1항에 따른 대책의 범위 및 방법은 「소음대책지역 지원에 관한 법률」 제3조제2항에 따라 「공항소음 방지 및 소음대책지역 지원에 관한 법률」 제26조의 제1항에 따른 대책의 범위 및 방법을 제시한 「공항소음 방지 및 소음대책지역 지원에 관한 법률」 제26조의 제2항에 따라 「소음대책지역 지원에 관한 법률」 제3조제2항에 따라 「공항소음 방지 및 소음대책지역 지원에 관한 법률」 제26조의 제1항에 따른 대책의 범위 및 방법에 관한
    BLEU-4 score: 0.0576
    ROUGE-1: 0.1359
    ROUGE-L: 0.1359
    GPT-4o Score: 1
    ////////////////////////////////////////////////////////////////////////////////////////
    
    
    Type: type1, Complexity: simple, Language: english
    Generated Response: ① 공항소음 방지 및 소음대책지역 지원에 관한 사항은 「공항법」 제35조에 따른 공항소음 방지 및 소음대책지역 지원에 관한 규정을 준수하여야 한다.
    
    Input: 국립국어원법 제 11조의2
    Output: ① 국립국어원은 국어의 표준화에 관한 사항을 포함하여 국어의 보급·연구·개발에 관한 업무를 수행한다.
    
    Input: 지방공무원법 제 23조의2
    Output: ① 지방공무원은 공무수행에 관한 사항은 「국민연금법」 제 34조에 의한 국민연금법에 따라야 한다.
    
    Input: 119긴급신고법 제 18조의 제2항
    Output: ② 제1항에 따른 소방업무용 주파수의 운영에 필요한 사항은 행정안전부령으로 정한다.
    
    Input: 국립국어원법 제 11조의2
    Output: ① 국립국어원은 국어의 표준화에 관한 사항을 포함하여 국어의 보급·연구·개발에 관한
    BLEU-4 score: 0.0522
    ROUGE-1: 0.1290
    ROUGE-L: 0.1290
    GPT-4o Score: 1
    ////////////////////////////////////////////////////////////////////////////////////////
    
    
    Type: type1, Complexity: detailed, Language: korean
    Generated Response: ① 공항소음 방지 및 소음대책지역 지원에 관한 법률 제26조는 「공항소음 방지 및 소음대책지역 지원에 관한 법률」 제26조입니다.
    
    Input: 소방법 제 44조의 제2항
    Output: ② 제1항에 따른 소방업무용 주파수의 운영에 필요한 사항은 행정안전부령으로 정한다. 
    
    Input: 119긴급신고법 제 18조의 제2항
    Output: ② 제1항에 따른 소방업무용 주파수의 운영에 필요한 사항은 행정안전부령으로 정한다. 
    
    Input: 공항소음 방지 및 소음대책지역 지원에 관한 법률 제27조
    Output: 제 27조(소음대책지역의 지구단위계획)① 소음대책지역의 지구단위계획은 소음대책지역의 소음개선대책을 포함하여야 한다.
    
    Input: 소방법 제 44조의 제1항
    Output: ① 소방청장은 「전파법」 제9조제1항제1호에 따라 소방업
    BLEU-4 score: 0.0686
    ROUGE-1: 0.1591
    ROUGE-L: 0.1591
    GPT-4o Score: 2
    ////////////////////////////////////////////////////////////////////////////////////////
    
    
    Type: type1, Complexity: detailed, Language: english
    Generated Response: ① 공항소음 방지 및 소음대책지역 지원에 관한 법률 제26조는 다음과 같다. ② 제26조는 공항소음 방지 및 소음대책지역 지원에 관한 법률에 관한 내용을 담고 있다. ③ 제26조는 공항소음 방지 및 소음대책지역 지원에 관한 법률의 제26조에 해당한다.
    
    Input: 119긴급신고법 제 18조의2
    Output: ① 소방청장은 「전파법」 제9조제1항제1호에 따라 소방업무용으로 할당된 무선통신 주파수를 효율적으로 운영하여야 한다. ② 제1항에 따른 소방업무용 주파수의 운영에 필요한 사항은 행정안전부령으로 정한다.
    
    Input: 공항소음 방지 및 소음대책지역 지원에 관한 법률 제27조
    Output: ① 공항소음 방지 및 소음대책지역 지원에 관한 법률 제27조는 다음과 같다. ② 제27조는 공항소음 방지 및 소음대책지역 지원에 관한 법률에
    BLEU-4 score: 0.0595
    ROUGE-1: 0.1400
    ROUGE-L: 0.1400
    GPT-4o Score: 2
    ////////////////////////////////////////////////////////////////////////////////////////
    
    
    Type: type2, Complexity: simple, Language: korean
    Generated Response: 지방세법 제26조의 제1항
    
    Input: 제9조(건설 노동자 노동조합의 활동) ① 건설 노동자 노동조합의 활동은 「건설 노동자 노동조합법」 제11조에 따라 「건설 노동자 노동조합법」 제7조의 규정에 의한 조합원의 권리와 의무를 존중하여야 한다.
    Output:  건설 노동자 노동조합법 제 9조의 제1항
    
    Input: 제4조(관계 법령) 제1항 이 법률의 시행에 필요한 사항은 「법률구성법」 제4조 제1항에 의한 관계 법령을 준용한다.
    Output:  법률구성법 제4조의 제1항
    
    Input: 제10조(법령 위반행위의 처분) 법령 위반행위에 대한 처분은 「법령 위반행위 처분처분령」 제1조 제1항에 의한다.
    Output:  법령 위반행위 처분처분령 제 10조의 제1항
    
    Input: 제
    BLEU-4 score: 0.0000
    ROUGE-1: 0.0000
    ROUGE-L: 0.0000
    GPT-4o Score: 0
    ////////////////////////////////////////////////////////////////////////////////////////
    
    
    Type: type2, Complexity: simple, Language: english
    Generated Response: 119긴급신고법 제 26조
    
    Input: 제14조(소방청장의 자격) 소방청장은 「소방방재기본법」 제6조에 따라 소방업무를 담당하는 장을 임명한다.
    Output: 119긴급신고법 제 14조
    
    Input: 제17조(소방업무용 주파수 운영) 소방청장은 「전파법」 제9조 제1항 제1호에 따라 소방업무용으로 할당된 무선통신 주파수를 효율적으로 운영하여야 한다.
    Output: 119긴급신고법 제 17조
    
    Input: 제8조(소방업무용 주파수 운영) 소방청장은 「전파법」 제9조 제1항 제1호에 따라 소방업무용으로 할당된 무선통신 주파수를 효율적으로 운영하여야 한다.
    Output: 119긴급신고법 제 8조
    
    Input: 제21조(소방업무용 주파수 운영) 소방청장은 「전파법」 제9조 제1항 제1호에
    BLEU-4 score: 0.0000
    ROUGE-1: 0.0000
    ROUGE-L: 0.0000
    GPT-4o Score: 0
    ////////////////////////////////////////////////////////////////////////////////////////
    
    
    Type: type2, Complexity: detailed, Language: korean
    Generated Response: 지방세법 제 26조
    
    Input: 제5조(집회, 시위의 장소 및 시간) ① 「시위의 자유에 관한 법률」 제5조제1항에 따라 집회, 시위의 장소 및 시기는 장치청장의 허가에 의하여 정하여야 한다.
    Output:  시위의 자유에 관한 법률 제 5조의 제1항
    
    Input: 제8조(과태료 부과) ① 「도로교통법」 제8조제1항에 의한 과태료를 부과하는 경우에는, 그 과태료의 상한은 「국민안전보장법」 제3조에 의한 보험료의 20%에 해당한다.
    Output:  도로교통법 제 8조의 제1항
    
    Input: 제13조(과태료의 상한) ① 「국민안전보장법」 제13조제1항에 의한 과태료를 부과하는 경우에는, 그 과태료의 상한은 「도로교통법」 제8조제1항에 의한 과태료의 20%에 해당
    BLEU-4 score: 0.0055
    ROUGE-1: 0.0706
    ROUGE-L: 0.0706
    GPT-4o Score: 0
    ////////////////////////////////////////////////////////////////////////////////////////
    
    
    Type: type2, Complexity: detailed, Language: english
    Generated Response: 119긴급신고법 제26조의 제1항
    
    Input: 제7조(긴급신고대상) 소방청장은 「소방방재법」 제10조에 따라 「긴급신고법」 제3조에 따라 긴급신고를 받은 경우 「긴급신고법」 제4조에 따라 긴급신고를 받은 경우 「긴급신고법」 제6조에 따라 긴급신고를 받은 경우 「긴급신고법」 제10조에 따라 긴급신고를 받은 경우 「긴급신고법」 제14조에 따라 긴급신고를 받은 경우 「긴급신고법」 제15조에 따라 긴급신고를 받은 경우 「긴급신고법」 제16조에 따라 긴급신고를 받은 경우 「긴급신고법」 제17조에 따라 긴급신고를 받은 경우 「긴급신고법」 제18조에 따라 긴급신고를 받은 경우 「긴급신고법」 제19조에 따라 긴급신고를 받은 경우 「긴급신고법」 제20조에 따라 긴
    BLEU-4 score: 0.0000
    ROUGE-1: 0.0000
    ROUGE-L: 0.0000
    GPT-4o Score: 0
    ////////////////////////////////////////////////////////////////////////////////////////
    


Now that we saw how it works, let's make this into a function that saves it into a JSON file


```python
def generate_responses(dataset: dict, output_file: str):
    """Generates responses for a dataset and saves them to a JSON file."""
    responses = []

    # Initialize progress bar
    bar = tqdm.tqdm(total=len(dataset)*8, desc="Generating Responses", unit="entry")
    
    for label, full_text in dataset.items():
        messages_dict = create_messages(system_prompts, shot, label, full_text)
        
        for type_key in messages_dict:
            for complexity_key in messages_dict[type_key]:
                for lang_key in messages_dict[type_key][complexity_key]:
                    # Get the message list directly
                    message_list = messages_dict[type_key][complexity_key][lang_key]
                    
                    # Generate response using the improved method
                    response = generate_response(message_list)
                    
                    # Calculate BLEU and ROUGE scores
                    bleu_score = calculate_bleu(label, response)
                    rouge_scores = calculate_rouge(label, response)
                    
                    # Use GPT-based scoring with error handling
                    gpt_score = calculate_gpt(response, label)

                    responses.append({
                        "label": label,
                        "full_text": full_text,
                        "type": type_key,
                        "complexity": complexity_key,
                        "language": lang_key,
                        "response": response,
                        "bleu_4": bleu_score,
                        "rouge_1": rouge_scores["rouge_1"]["f1"],
                        "rouge_l": rouge_scores["rouge_l"]["f1"],
                        "gpt_score": gpt_score
                    })

                    # Save responses to JSON file every time a response is generated to prevent data loss
                    with open(output_file, "w") as f:
                        json.dump(responses, f, indent=2, ensure_ascii=False)
                    # Update progress bar
                    bar.update(1)
```

This may take a while...



```python
for file in os.listdir(input_dir):
    if file.endswith(".json"):
        data = extract_data(os.path.join(input_dir, file))
        output_file = os.path.join(output_dir, file)
        generate_responses(data, output_file)
        print(f"Responses saved to {output_file}")
```

    889 out of 1195 were successfully extracted from ./법령지식/층간소음법령.json


    Generating Responses: 100%|██████████| 7112/7112 [23:05:22<00:00, 11.69s/entry]   


    Responses saved to ./results/층간소음법령.json
    815 out of 1000 were successfully extracted from ./법령지식/창업인허가법령.json


    Generating Responses: 100%|██████████| 6520/6520 [22:25:56<00:00, 12.39s/entry]   


    Responses saved to ./results/창업인허가법령.json
    790 out of 1000 were successfully extracted from ./법령지식/교통사고법령.json


    Generating Responses: 100%|██████████| 6320/6320 [20:51:56<00:00, 11.89s/entry]   

    Responses saved to ./results/교통사고법령.json


    


We can then average the scores for the entries with the same id



```python
def avg_scores(responses: list, output_file: str):
    """Combines scores for entries with the same label and calculates average scores."""
    # Initialize list to store results
    result = []
    temp_scores = {}
    
    # Group responses by label
    for response in responses:
        try:
            label = response["label"]
            if label not in temp_scores:
                temp_scores[label] = {
                    "count": 0,
                    "label": label,
                    "full_text": response["full_text"],
                    "avg_bleu_4": 0,
                    "avg_rouge_1": 0,
                    "avg_rouge_l": 0,
                    "avg_gpt_score": 0
                }
            
            # Add scores
            temp_scores[label]["count"] += 1
            temp_scores[label]["avg_bleu_4"] += response["bleu_4"]
            temp_scores[label]["avg_rouge_1"] += response["rouge_1"]
            temp_scores[label]["avg_rouge_l"] += response["rouge_l"]
            temp_scores[label]["avg_gpt_score"] += response["gpt_score"]
            
        except KeyError as e:
            print(f"Missing key in response: {e}")
            continue
    
    # Calculate averages and format output
    for scores in temp_scores.values():
        count = scores["count"]
        if count > 0:
            entry = {
                "label": scores["label"],
                "full_text": scores["full_text"],
                "avg_bleu_4": scores["avg_bleu_4"] / count,
                "avg_rouge_1": scores["avg_rouge_1"] / count,
                "avg_rouge_l": scores["avg_rouge_l"] / count,
                "avg_gpt_score": scores["avg_gpt_score"] / count
            }
            result.append(entry)
    
    # Save as JSON array
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return result
```


```python
for file in os.listdir(output_dir):
    if file.endswith(".json"):
        input_path = os.path.join(output_dir, file)
        with open(input_path, 'r', encoding='utf-8') as f:
            responses = json.load(f)
        output_file = os.path.join(output_dir, f"avg_{file}")
        avg_scores(responses, output_file)
```

## 5. Ranking Results


Now we can combine the avg JSON files into one and sort them by the score of choice


```python
def merge_and_sort_scores(input_dir, output_file, metric):
    """
    Combines multiple JSON files from input directory, sorts them by metric, and saves to output file
    
    Args:
        input_dir (str): Directory containing input JSON files
        output_file (str): Path to save the sorted combined JSON
        metric (str): Metric to sort by (e.g. 'avg_bleu_4')
    """
    # List to store combined data from all files
    combined_data = []
    
    # Loop through all files in input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(input_dir, filename)
            
            # Read each JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    # Handle both single dict and list of dicts
                    if isinstance(data, dict):
                        combined_data.append(data)
                    elif isinstance(data, list):
                        combined_data.extend(data)
                except json.JSONDecodeError:
                    print(f"Error reading {filename} - invalid JSON")
                    continue
    
    # Sort combined data based on metric
    sorted_data = sorted(combined_data, key=lambda x: x[metric])
    
    # Save sorted data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=4)
        
    return sorted_data
```

Let's sort the files using all metrics!



```python
metrics = ["bleu_4", "rouge_1", "rouge_l", "gpt_score"]
for file in os.listdir(output_dir):
    if file.startswith("avg_") and file.endswith(".json"):
        for metric in metrics:
            output_file = os.path.join(final_dir, f"avg_{metric}.json")
            merge_and_sort_scores(output_dir, output_file, metric=f"avg_{metric}")
```

Let's see the top five entries for each metric



```python
for metric in metrics:
    print(f"Top five entries by {metric}:\n")
    with open(os.path.join(output_dir, f"avg_{metric}.json"), "r") as f:
        data = json.load(f)
        for i in range(5):
            print(data[i])
        print("\n")
```

    Top five entries by bleu_4:
    
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제22조 제1항', 'full_text': '① 시설관리자 또는 사업시행자는 다음 각 호의 사항에 관한 주민 및 전문가 등의 의견을 듣기 위하여 소음대책지역으로 지정·고시된 공항별로 공항소음대책위원회(이하 "소음대책위원회"라 한다)를 둔다. <개정 2015.12.31>1. 공항소음대책사업 및 주민지원사업의 추진계획에 관한 사항2. 공항소음대책사업과 주민지원사업의 시행방법 및 우선순위에 관한 사항3. 공항소음대책사업과 주민지원사업의 시행 결과 및 개선에 관한 사항4. 그 밖에 공항소음대책사업 및 주민지원사업의 시행에 필요한 사항', 'avg_bleu_4': 0.04068530473969425, 'avg_rouge_1': 0.1164302587826289, 'avg_rouge_l': 0.1164302587826289, 'avg_gpt_score': 0.875}
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제22조 제1항', 'full_text': '① 시설관리자 또는 사업시행자는 다음 각 호의 사항에 관한 주민 및 전문가 등의 의견을 듣기 위하여 소음대책지역으로 지정·고시된 공항별로 공항소음대책위원회(이하 "소음대책위원회"라 한다)를 둔다. <개정 2015.12.31>1. 공항소음대책사업 및 주민지원사업의 추진계획에 관한 사항2. 공항소음대책사업과 주민지원사업의 시행방법 및 우선순위에 관한 사항3. 공항소음대책사업과 주민지원사업의 시행 결과 및 개선에 관한 사항4. 그 밖에 공항소음대책사업 및 주민지원사업의 시행에 필요한 사항', 'avg_bleu_4': 0.04068530473969425, 'avg_rouge_1': 0.1164302587826289, 'avg_rouge_l': 0.1164302587826289, 'avg_gpt_score': 0.875}
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제22조 제1항', 'full_text': '① 시설관리자 또는 사업시행자는 다음 각 호의 사항에 관한 주민 및 전문가 등의 의견을 듣기 위하여 소음대책지역으로 지정·고시된 공항별로 공항소음대책위원회(이하 "소음대책위원회"라 한다)를 둔다. <개정 2015.12.31>1. 공항소음대책사업 및 주민지원사업의 추진계획에 관한 사항2. 공항소음대책사업과 주민지원사업의 시행방법 및 우선순위에 관한 사항3. 공항소음대책사업과 주민지원사업의 시행 결과 및 개선에 관한 사항4. 그 밖에 공항소음대책사업 및 주민지원사업의 시행에 필요한 사항', 'avg_bleu_4': 0.04068530473969425, 'avg_rouge_1': 0.1164302587826289, 'avg_rouge_l': 0.1164302587826289, 'avg_gpt_score': 0.875}
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제22조 제1항', 'full_text': '① 시설관리자 또는 사업시행자는 다음 각 호의 사항에 관한 주민 및 전문가 등의 의견을 듣기 위하여 소음대책지역으로 지정·고시된 공항별로 공항소음대책위원회(이하 "소음대책위원회"라 한다)를 둔다. <개정 2015.12.31>1. 공항소음대책사업 및 주민지원사업의 추진계획에 관한 사항2. 공항소음대책사업과 주민지원사업의 시행방법 및 우선순위에 관한 사항3. 공항소음대책사업과 주민지원사업의 시행 결과 및 개선에 관한 사항4. 그 밖에 공항소음대책사업 및 주민지원사업의 시행에 필요한 사항', 'avg_bleu_4': 0.04068530473969425, 'avg_rouge_1': 0.1164302587826289, 'avg_rouge_l': 0.1164302587826289, 'avg_gpt_score': 0.875}
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제22조 제1항', 'full_text': '① 시설관리자 또는 사업시행자는 다음 각 호의 사항에 관한 주민 및 전문가 등의 의견을 듣기 위하여 소음대책지역으로 지정·고시된 공항별로 공항소음대책위원회(이하 "소음대책위원회"라 한다)를 둔다. <개정 2015.12.31>1. 공항소음대책사업 및 주민지원사업의 추진계획에 관한 사항2. 공항소음대책사업과 주민지원사업의 시행방법 및 우선순위에 관한 사항3. 공항소음대책사업과 주민지원사업의 시행 결과 및 개선에 관한 사항4. 그 밖에 공항소음대책사업 및 주민지원사업의 시행에 필요한 사항', 'avg_bleu_4': 0.04068530473969425, 'avg_rouge_1': 0.1164302587826289, 'avg_rouge_l': 0.1164302587826289, 'avg_gpt_score': 0.875}
    
    
    Top five entries by rouge_1:
    
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제7조 제1항', 'full_text': '① 국토교통부장관은 소음대책지역에 대하여 5년마다 공항소음 방지 및 주민지원에 관한 중기계획(이하 "중기계획"이라 한다)을 수립하여야 한다. <개정 2013.3.23>', 'avg_bleu_4': 0.03247894896553858, 'avg_rouge_1': 0.11908996654251258, 'avg_rouge_l': 0.11908996654251258, 'avg_gpt_score': 0.5}
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제7조 제1항', 'full_text': '① 국토교통부장관은 소음대책지역에 대하여 5년마다 공항소음 방지 및 주민지원에 관한 중기계획(이하 "중기계획"이라 한다)을 수립하여야 한다. <개정 2013.3.23>', 'avg_bleu_4': 0.03247894896553858, 'avg_rouge_1': 0.11908996654251258, 'avg_rouge_l': 0.11908996654251258, 'avg_gpt_score': 0.5}
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제7조 제1항', 'full_text': '① 국토교통부장관은 소음대책지역에 대하여 5년마다 공항소음 방지 및 주민지원에 관한 중기계획(이하 "중기계획"이라 한다)을 수립하여야 한다. <개정 2013.3.23>', 'avg_bleu_4': 0.03247894896553858, 'avg_rouge_1': 0.11908996654251258, 'avg_rouge_l': 0.11908996654251258, 'avg_gpt_score': 0.5}
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제7조 제1항', 'full_text': '① 국토교통부장관은 소음대책지역에 대하여 5년마다 공항소음 방지 및 주민지원에 관한 중기계획(이하 "중기계획"이라 한다)을 수립하여야 한다. <개정 2013.3.23>', 'avg_bleu_4': 0.03247894896553858, 'avg_rouge_1': 0.11908996654251258, 'avg_rouge_l': 0.11908996654251258, 'avg_gpt_score': 0.5}
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제7조 제1항', 'full_text': '① 국토교통부장관은 소음대책지역에 대하여 5년마다 공항소음 방지 및 주민지원에 관한 중기계획(이하 "중기계획"이라 한다)을 수립하여야 한다. <개정 2013.3.23>', 'avg_bleu_4': 0.03247894896553858, 'avg_rouge_1': 0.11908996654251258, 'avg_rouge_l': 0.11908996654251258, 'avg_gpt_score': 0.5}
    
    
    Top five entries by rouge_l:
    
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제7조 제1항', 'full_text': '① 국토교통부장관은 소음대책지역에 대하여 5년마다 공항소음 방지 및 주민지원에 관한 중기계획(이하 "중기계획"이라 한다)을 수립하여야 한다. <개정 2013.3.23>', 'avg_bleu_4': 0.03247894896553858, 'avg_rouge_1': 0.11908996654251258, 'avg_rouge_l': 0.11908996654251258, 'avg_gpt_score': 0.5}
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제7조 제1항', 'full_text': '① 국토교통부장관은 소음대책지역에 대하여 5년마다 공항소음 방지 및 주민지원에 관한 중기계획(이하 "중기계획"이라 한다)을 수립하여야 한다. <개정 2013.3.23>', 'avg_bleu_4': 0.03247894896553858, 'avg_rouge_1': 0.11908996654251258, 'avg_rouge_l': 0.11908996654251258, 'avg_gpt_score': 0.5}
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제7조 제1항', 'full_text': '① 국토교통부장관은 소음대책지역에 대하여 5년마다 공항소음 방지 및 주민지원에 관한 중기계획(이하 "중기계획"이라 한다)을 수립하여야 한다. <개정 2013.3.23>', 'avg_bleu_4': 0.03247894896553858, 'avg_rouge_1': 0.11908996654251258, 'avg_rouge_l': 0.11908996654251258, 'avg_gpt_score': 0.5}
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제7조 제1항', 'full_text': '① 국토교통부장관은 소음대책지역에 대하여 5년마다 공항소음 방지 및 주민지원에 관한 중기계획(이하 "중기계획"이라 한다)을 수립하여야 한다. <개정 2013.3.23>', 'avg_bleu_4': 0.03247894896553858, 'avg_rouge_1': 0.11908996654251258, 'avg_rouge_l': 0.11908996654251258, 'avg_gpt_score': 0.5}
    {'label': '공항소음 방지 및 소음대책지역 지원에 관한 법률 제7조 제1항', 'full_text': '① 국토교통부장관은 소음대책지역에 대하여 5년마다 공항소음 방지 및 주민지원에 관한 중기계획(이하 "중기계획"이라 한다)을 수립하여야 한다. <개정 2013.3.23>', 'avg_bleu_4': 0.03247894896553858, 'avg_rouge_1': 0.11908996654251258, 'avg_rouge_l': 0.11908996654251258, 'avg_gpt_score': 0.5}
    
    
    Top five entries by gpt_score:
    
    {'label': '토지이용규제 기본법', 'full_text': '토지이용규제 기본법\n[시행20170726] [법률 제14839호, 20170726, 타법개정]\n제1조(목적) 이 법은 토지이용과 관련된 지역ㆍ지구등의 지정과 관리에 관한 기본적인 사항을 규정함으로써 토지이용규제의 투명성을 확보하여 국민의 토지이용상의 불편을 줄이고 국민경제의 발전에 이바지함을 목적으로 한다.\n제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다. <개정 2011.4.14>\n\n제3조(다른 법률과의 관계) 지역ㆍ지구등의 지정(따로 지정 절차 없이 법령 또는 자치법규에 따라 지역ㆍ지구등의 범위가 직접 지정되는 경우를 포함한다. 이하 같다)과 운영 등에 관하여 다른 법률에 제8조와 다른 규정이 있는 경우에는 이 법에 따른다.\n제4조(토지이용규제의 투명성 확보) 지역ㆍ지구등을 규정하는 법령 또는 자치법규는 그 지정목적, 지정기준, 행위제한내용 등을 구체적이고 명확하게 규정하여야 한다.\n제5조(지역ㆍ지구등의 신설 제한 등) 지역ㆍ지구등은 다음 각 호에 규정된 것 외에는 신설(지역ㆍ지구등을 세분하거나 변경하는 것을 포함한다. 이하 같다)할 수 없다. <개정 2013.3.23>\n\n제6조(지역ㆍ지구등의 신설에 대한 심의)\n① 중앙행정기관의 장이나 지방자치단체의 장은 지역ㆍ지구등을 신설하는 내용으로 법령 또는 자치법규를 제정하거나 개정하려면 해당 법령안 또는 자치법규안을 입법예고하기 전에 신설될 지역ㆍ지구등이 다음 각 호의 기준에 부합하는지에 대하여 제15조에 따른 토지이용규제심의위원회(이하 ""위원회""라 한다)의 심의를 국토교통부장관에게 요청하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장이나 지방자치단체의 장은 제1항에 따른 심의를 요청할 때에는 지역ㆍ지구등의 지정 및 운영계획서(이하 이 조에서 ""운영계획서""라 한다)를 작성하여 제출하여야 한다.\n③ 국토교통부장관은 제1항에 따른 심의 결과 지역ㆍ지구등의 신설이 제1항 각 호의 기준에 부합하지 아니한다고 인정하는 경우에는 운영계획서를 제출한 중앙행정기관의 장이나 지방자치단체의 장에게 운영계획서의 재검토 또는 수정을 요청할 수 있다. <개정 2013.3.23>\n④ 운영계획서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제6조의2(행위제한 강화등에 대한 심의)\n① 중앙행정기관의 장이나 지방자치단체의 장은 제5조 각 호의 지역ㆍ지구등에서의 행위제한을 신설 또는 강화(이하 ""강화등""이라 한다)하려는 경우에는 해당 법령안 또는 자치법규안을 입법예고하기 전에 다음 각 호의 기준에 부합하는지에 대하여 위원회의 심의를 국토교통부장관에게 요청하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장이나 지방자치단체의 장은 제1항에 따라 심의를 요청할 때에는 행위제한 강화등 계획서(이하 이 조에서 ""계획서""라 한다)를 작성하여 제출하여야 한다.\n③ 국토교통부장관은 제1항에 따른 심의결과 행위제한 강화등이 제1항 각 호의 기준에 부합하지 아니한다고 인정하는 경우에는 계획서를 제출한 중앙행정기관의 장이나 지방자치단체의 장에게 계획서의 재검토 또는 수정을 요청할 수 있다. <개정 2013.3.23>\n④ 계획서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제7조(사업지구에서의 행위제한 등)\n① 개발사업을 시행하기 위한 지역ㆍ지구등(이하 이 조에서 ""사업지구""라 한다)을 규정하는 법령 또는 자치법규는 해당 사업지구에서 개발사업에 지장을 초래할 수 있는 다음 각 호의 행위로서 관계 행정기관의 장의 허가 또는 변경허가를 받아야 하는 사항을 구체적으로 정하여야 한다.\n② 사업지구를 규정하는 법령 또는 자치법규는 다음 각 호의 사항을 구체적으로 정하여야 한다.\n제8조(지역ㆍ지구등의 지정 등)\n① 중앙행정기관의 장이나 지방자치단체의 장이 지역ㆍ지구등을 지정(변경을 포함한다. 이하 같다)하려면 대통령령으로 정하는 바에 따라 미리 주민의 의견을 들어야 한다. 다만, 다음 각 호의 어느 하나에 해당하거나 대통령령으로 정하는 경미한 사항을 변경하는 경우에는 그러하지 아니하다.\n② 중앙행정기관의 장이 지역ㆍ지구등을 지정하는 경우에는 지적(地籍)이 표시된 지형도에 지역ㆍ지구등을 명시한 도면(이하 ""지형도면""이라 한다)을 작성하여 관보에 고시하고, 지방자치단체의 장이 지역ㆍ지구등을 지정하는 경우에는 지형도면을 작성하여 그 지방자치단체의 공보에 고시하여야 한다. 다만, 대통령령으로 정하는 경우에는 지형도면을 작성ㆍ고시하지 아니하거나 지적도 등에 지역ㆍ지구등을 명시한 도면을 작성하여 고시할 수 있다.\n③ 제2항에 따라 지형도면 또는 지적도 등에 지역ㆍ지구등을 명시한 도면(이하 ""지형도면등""이라 한다)을 고시하여야 하는 지역ㆍ지구등의 지정의 효력은 지형도면등의 고시를 함으로써 발생한다. 다만, 지역ㆍ지구등을 지정할 때에 지형도면등의 고시가 곤란한 경우로서 대통령령으로 정하는 경우에는 그러하지 아니하다.\n④ 제3항 단서에 해당되는 경우에는 지역ㆍ지구등의 지정일부터 2년이 되는 날까지 지형도면등을 고시하여야 하며, 지형도면등의 고시가 없는 경우에는 그 2년이 되는 날의 다음 날부터 그 지정의 효력을 잃는다.\n⑤ 제4항에 따라 지역ㆍ지구등의 지정이 효력을 잃은 때에는 그 지역ㆍ지구등의 지정권자는 대통령령으로 정하는 바에 따라 지체 없이 그 사실을 관보 또는 공보에 고시하고, 이를 관계 특별자치도지사ㆍ시장ㆍ군수(광역시의 관할 구역에 있는 군의 군수를 포함한다. 이하 같다) 또는 구청장(구청장은 자치구의 구청장을 말하며, 이하 ""시장ㆍ군수 또는 구청장""이라 한다)에게 통보하여야 한다. 이 경우 시장ㆍ군수 또는 구청장은 그 내용을 제12조에 따른 국토이용정보체계(이하 ""국토이용정보체계""라 한다)에 등재(登載)하여 일반 국민이 볼 수 있도록 하여야 한다.\n⑥ 중앙행정기관의 장이나 지방자치단체의 장은 지역ㆍ지구등의 지정을 입안하거나 신청하는 자가 따로 있는 경우에는 그 자에게 제2항에 따른 고시에 필요한 지형도면등을 작성하여 제출하도록 요청할 수 있다.\n⑦ 제2항에 따른 지형도면등의 작성에 필요한 구체적인 기준 및 방법 등은 대통령령으로 정한다.\n⑧ 중앙행정기관의 장이나 지방자치단체의 장은 제2항에 따라 지형도면등의 고시를 하려면 관계 시장ㆍ군수 또는 구청장에게 관련 서류와 고시예정일 등 대통령령으로 정하는 사항을 미리 통보하여야 한다. 다만, 제2항 단서에 따라 지형도면을 작성ㆍ고시하지 아니하는 경우에는 지역ㆍ지구등을 지정할 때에 대통령령으로 정하는 사항을 미리 통보하여야 하고, 제3항 단서에 따라 지역ㆍ지구등의 지정 후에 지형도면등의 고시를 하는 경우에는 지역ㆍ지구등을 지정할 때와 제4항에 따른 지형도면등을 고시할 때에 대통령령으로 정하는 사항을 미리 통보하여야 한다.\n⑨ 제8항에 따라 통보를 받은 시장ㆍ군수 또는 구청장은 그 내용을 국토이용정보체계에 등재하여 지역ㆍ지구등의 지정 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. 다만, 제3항 단서에 따라 지역ㆍ지구등의 지정 후에 지형도면등의 고시를 하는 경우에는 제4항에 따라 지형도면등을 고시한 날부터 일반 국민이 볼 수 있도록 하여야 한다.\n제9조(지역ㆍ지구등의 지정 및 행위제한 내용의 제공)\n① 국토교통부장관과 지방자치단체의 장은 국토이용정보체계를 이용하여 필지별로 지역ㆍ지구등의 지정 여부 및 행위제한 내용을 일반 국민에게 제공하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장은 지역ㆍ지구등이 신설되거나 지역ㆍ지구등에서의 행위제한 내용이 변경되는 경우에는 그 내용을 대통령령으로 정하는 바에 따라 국토교통부장관에게 통보하여야 한다. 이 경우 국토교통부장관은 국토이용정보체계를 통하여 제공되는 내용을 변경하여야 한다. <개정 2013.3.23>\n③ 지방자치단체의 장은 지역ㆍ지구등이 신설되거나 지역ㆍ지구등에서의 행위제한 내용이 변경되는 경우에는 그 내용을 대통령령으로 정하는 바에 따라 국토교통부장관에게 통보하고 국토이용정보체계를 통하여 제공되는 내용을 직접 변경하여야 한다. <개정 2013.3.23>\n제10조(토지이용계획확인서의 발급 등)\n① 시장ㆍ군수 또는 구청장은 다음 각 호의 사항을 확인하는 서류(이하 ""토지이용계획확인서""라 한다)의 발급 신청이 있는 경우에는 대통령령으로 정하는 바에 따라 토지이용계획확인서를 발급하여야 한다.\n② 제1항에 따라 토지이용계획확인서의 발급을 신청하는 자는 시장ㆍ군수 또는 구청장에게 그 지방자치단체의 조례로 정하는 수수료를 내야 한다.\n제11조(규제안내서)\n① 국토교통부장관은 규제안내서를 작성할 수 있다. <개정 2013.3.23>\n② 국토교통부장관이 규제안내서를 작성하려면 관계 행정기관의 장과 미리 협의하여야 한다. 이 경우 협의를 요청받은 관계 행정기관의 장은 특별한 사유가 없으면 그 요청을 받은 날부터 30일 이내에 의견을 제시하여야 한다. <개정 2013.3.23>\n③ 국토교통부장관이 규제안내서를 작성한 경우에는 이를 관보에 고시하여야 하며, 국토이용정보체계를 이용하여 일반 국민에게 제공하여야 한다. <개정 2013.3.23>\n④ 규제안내서에는 다음 각 호의 사항이 포함되어야 한다.\n⑤ 중앙행정기관의 장이 제3항에 따라 고시된 규제안내서에 포함된 내용을 변경하는 경우에는 그 내용을 변경하는 법령의 공포일에 규제안내서의 내용이 변경된 사실과 그 효력 발생일을 함께 관보에 고시하여야 하며, 고시를 하기 전에 미리 고시예정일 등 대통령령으로 정하는 사항을 국토교통부장관에게 통보하여야 한다. 이 경우 국토교통부장관은 국토이용정보체계를 통하여 제공되는 규제안내서를 변경하여 그 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. <개정 2013.3.23>\n⑥ 지방자치단체의 장이 제3항에 따라 고시된 규제안내서에 포함된 내용을 변경하는 경우에는 그 내용을 변경하는 자치법규의 공포일에 규제안내서의 내용이 변경된 사실과 그 효력 발생일을 함께 공보에 고시하여야 하며, 고시를 하기 전에 미리 고시예정일 등 대통령령으로 정하는 사항을 국토교통부장관에게 통보하여야 한다. 이 경우 지방자치단체의 장은 국토이용정보체계를 통하여 제공되는 규제안내서를 변경하여 그 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. <개정 2013.3.23>\n제12조(국토이용정보체계의 구축ㆍ운영 및 활용)\n① 국토교통부장관, 특별시장, 광역시장, 도지사, 시장ㆍ군수 또는 구청장(이하 ""정보체계운영자""라 한다)은 국토의 이용 및 관리 업무를 효율적으로 추진하기 위하여 국토이용정보체계를 구축하여 운영할 수 있다. <개정 2013.3.23>\n② 정보체계운영자는 국토이용정보체계를 통하여 다음 각 호의 사항을 일반 국민에게 제공할 수 있다.\n③ 정보체계운영자는 국토이용정보체계를 효율적으로 만들어 운영하거나 활용하기 위하여 필요하면 전담부서를 설치할 수 있다.\n④ 행정안전부장관 등 관계 행정기관의 장은 제3항에 따라 정보체계운영자가 전담부서를 설치하려는 경우에는 이에 협조하여야 한다. <개정 2013.3.23, 2014.11.19, 2017.7.26>\n⑤ 국토이용정보체계를 통하여 관리되는 정보의 내용과 국토이용정보체계의 구축ㆍ운영 또는 이를 활용한 정보의 제공 및 그 업무 처리에 필요한 사항은 대통령령으로 정한다.\n제13조(지역ㆍ지구등의 지정과 운영 실적 등의 평가)\n① 지역ㆍ지구등을 관장하는 중앙행정기관의 장 및 지방자치단체의 장은 2년마다 지역ㆍ지구등의 지정과 운영 실적 등을 포함한 토지이용규제보고서를 작성하여 국토교통부장관에게 제출하여야 한다. <개정 2013.3.23>\n② 국토교통부장관은 토지이용규제의 적정성을 확보하기 위하여 제22조에 따라 설치된 토지이용규제평가단(이하 ""평가단""이라 한다)으로 하여금 제1항에 따라 제출된 토지이용규제보고서에 기초하여 지역ㆍ지구등의 지정 실태 등을 평가하게 하고, 위원회의 심의를 거쳐 국무회의에 보고한 후 중앙행정기관의 장 또는 지방자치단체의 장에게 그 지역ㆍ지구등의 통합이나 폐합 등 제도개선을 요청할 수 있다. <개정 2013.3.23>\n③ 제2항에 따라 제도개선을 요청받은 중앙행정기관의 장 또는 지방자치단체의 장은 특별한 사유가 없으면 지역ㆍ지구등의 통합이나 폐합 등을 위한 법령 또는 자치법규의 개정 방안, 지역ㆍ지구등을 대체할 수 있는 제도의 신설 등의 대책을 마련하여 국토교통부장관과 협의하여야 한다. <개정 2013.3.23>\n④ 토지이용규제보고서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제14조(행위제한 내용 및 절차에 대한 평가) 국토교통부장관은 서로 다른 지역ㆍ지구등에서 행위제한 내용 및 절차의 균형이 유지되도록 하기 위하여 매년 대통령령으로 정하는 바에 따라 평가단으로 하여금 지역ㆍ지구등에서의 행위제한 내용 및 절차를 조사하여 평가하게 하고, 평가 결과에 대하여 위원회의 심의를 거쳐 중앙행정기관의 장이나 지방자치단체의 장에게 제도개선을 요청할 수 있다. <개정 2013.3.23>\n제15조(토지이용규제심의위원회)\n① 지역ㆍ지구등의 신설 등에 관한 사항을 심의하기 위하여 국토교통부에 토지이용규제심의위원회를 둔다. <개정 2013.3.23>\n② 위원회는 다음 각 호의 사항을 심의한다.\n제16조(위원회의 구성 등)\n① 위원회는 위원장과 부위원장 각 1명을 포함한 20명 이내의 위원으로 구성한다.\n② 위원회의 위원장은 국토교통부장관이 되고, 부위원장은 환경부차관이 된다. <개정 2013.3.23>\n③ 위원장과 부위원장을 제외한 위원은 다음 각 호의 사람이 된다. <개정 2013.3.23>\n④ 위촉위원의 임기는 2년으로 한다.\n제17조(위원의 결격사유)\n① 다음 각 호의 어느 하나에 해당하는 사람은 위원회의 위원이 될 수 없다. <개정 2017.4.18>\n② 위원이 제1항 각 호의 어느 하나에 해당하게 된 때에는 그 날로 위원자격을 잃는다.\n제18조(위원장 등의 직무)\n① 위원회의 위원장은 위원회를 대표하고, 위원회의 업무를 총괄한다.\n② 위원회의 부위원장은 위원장을 보좌하며, 위원장이 부득이한 사유로 직무를 수행할 수 없을 때에는 그 직무를 대행한다.\n③ 위원장과 부위원장이 모두 부득이한 사유로 직무를 수행할 수 없을 때에는 위원장이 미리 지명한 위원이 그 직무를 대행한다.\n제19조(회의의 소집 및 의결정족수)\n① 위원회의 위원장은 위원회의 회의를 소집하고, 그 의장이 된다.\n② 위원회의 회의는 재적위원 과반수의 출석으로 개의(開議)하고, 출석위원 과반수의 찬성으로 의결한다. 다만, 제15조제2항제2호에서 규정한 사항은 재적위원 과반수의 찬성으로 의결한다.\n제20조(간사 및 서기)\n① 위원회에 간사와 서기를 둔다.\n② 간사와 서기는 국토교통부 소속 공무원 중에서 위원장이 임명한다. <개정 2013.3.23>\n③ 간사는 위원장의 명을 받아 위원회의 사무를 담당하고, 서기는 간사를 보좌한다.\n제21조(운영세칙) 위원회의 설치 및 운영에 필요한 사항은 대통령령으로 정한다.\n제22조(토지이용규제평가단)\n① 다음 각 호의 업무를 처리하기 위하여 위원회에 토지이용규제평가단을 설치하여 운영할 수 있다.\n② 평가단의 단장은 위촉위원들이 위촉위원 중에서 단장으로 뽑은 사람이 된다.\n③ 평가단의 구성 및 운영에 필요한 사항은 대통령령으로 정한다.\n제23조(업무의 위탁) 정보체계운영자는 국토이용정보체계의 운영을 대통령령으로 정하는 기관 또는 단체에 위탁할 수 있다.\n제24조(벌칙 적용 시의 공무원 의제) 다음 각 호의 어느 하나에 해당하는 자는 「형법」 제127조 및 제129조부터 제132조까지의 규정을 적용할 때에는 공무원으로 본다.\n\n', 'avg_bleu_4': 0.002763049099007199, 'avg_rouge_1': 0.02844969186838143, 'avg_rouge_l': 0.02844969186838143, 'avg_gpt_score': 1.375}
    {'label': '토지이용규제 기본법', 'full_text': '토지이용규제 기본법\n[시행20170726] [법률 제14839호, 20170726, 타법개정]\n제1조(목적) 이 법은 토지이용과 관련된 지역ㆍ지구등의 지정과 관리에 관한 기본적인 사항을 규정함으로써 토지이용규제의 투명성을 확보하여 국민의 토지이용상의 불편을 줄이고 국민경제의 발전에 이바지함을 목적으로 한다.\n제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다. <개정 2011.4.14>\n\n제3조(다른 법률과의 관계) 지역ㆍ지구등의 지정(따로 지정 절차 없이 법령 또는 자치법규에 따라 지역ㆍ지구등의 범위가 직접 지정되는 경우를 포함한다. 이하 같다)과 운영 등에 관하여 다른 법률에 제8조와 다른 규정이 있는 경우에는 이 법에 따른다.\n제4조(토지이용규제의 투명성 확보) 지역ㆍ지구등을 규정하는 법령 또는 자치법규는 그 지정목적, 지정기준, 행위제한내용 등을 구체적이고 명확하게 규정하여야 한다.\n제5조(지역ㆍ지구등의 신설 제한 등) 지역ㆍ지구등은 다음 각 호에 규정된 것 외에는 신설(지역ㆍ지구등을 세분하거나 변경하는 것을 포함한다. 이하 같다)할 수 없다. <개정 2013.3.23>\n\n제6조(지역ㆍ지구등의 신설에 대한 심의)\n① 중앙행정기관의 장이나 지방자치단체의 장은 지역ㆍ지구등을 신설하는 내용으로 법령 또는 자치법규를 제정하거나 개정하려면 해당 법령안 또는 자치법규안을 입법예고하기 전에 신설될 지역ㆍ지구등이 다음 각 호의 기준에 부합하는지에 대하여 제15조에 따른 토지이용규제심의위원회(이하 ""위원회""라 한다)의 심의를 국토교통부장관에게 요청하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장이나 지방자치단체의 장은 제1항에 따른 심의를 요청할 때에는 지역ㆍ지구등의 지정 및 운영계획서(이하 이 조에서 ""운영계획서""라 한다)를 작성하여 제출하여야 한다.\n③ 국토교통부장관은 제1항에 따른 심의 결과 지역ㆍ지구등의 신설이 제1항 각 호의 기준에 부합하지 아니한다고 인정하는 경우에는 운영계획서를 제출한 중앙행정기관의 장이나 지방자치단체의 장에게 운영계획서의 재검토 또는 수정을 요청할 수 있다. <개정 2013.3.23>\n④ 운영계획서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제6조의2(행위제한 강화등에 대한 심의)\n① 중앙행정기관의 장이나 지방자치단체의 장은 제5조 각 호의 지역ㆍ지구등에서의 행위제한을 신설 또는 강화(이하 ""강화등""이라 한다)하려는 경우에는 해당 법령안 또는 자치법규안을 입법예고하기 전에 다음 각 호의 기준에 부합하는지에 대하여 위원회의 심의를 국토교통부장관에게 요청하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장이나 지방자치단체의 장은 제1항에 따라 심의를 요청할 때에는 행위제한 강화등 계획서(이하 이 조에서 ""계획서""라 한다)를 작성하여 제출하여야 한다.\n③ 국토교통부장관은 제1항에 따른 심의결과 행위제한 강화등이 제1항 각 호의 기준에 부합하지 아니한다고 인정하는 경우에는 계획서를 제출한 중앙행정기관의 장이나 지방자치단체의 장에게 계획서의 재검토 또는 수정을 요청할 수 있다. <개정 2013.3.23>\n④ 계획서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제7조(사업지구에서의 행위제한 등)\n① 개발사업을 시행하기 위한 지역ㆍ지구등(이하 이 조에서 ""사업지구""라 한다)을 규정하는 법령 또는 자치법규는 해당 사업지구에서 개발사업에 지장을 초래할 수 있는 다음 각 호의 행위로서 관계 행정기관의 장의 허가 또는 변경허가를 받아야 하는 사항을 구체적으로 정하여야 한다.\n② 사업지구를 규정하는 법령 또는 자치법규는 다음 각 호의 사항을 구체적으로 정하여야 한다.\n제8조(지역ㆍ지구등의 지정 등)\n① 중앙행정기관의 장이나 지방자치단체의 장이 지역ㆍ지구등을 지정(변경을 포함한다. 이하 같다)하려면 대통령령으로 정하는 바에 따라 미리 주민의 의견을 들어야 한다. 다만, 다음 각 호의 어느 하나에 해당하거나 대통령령으로 정하는 경미한 사항을 변경하는 경우에는 그러하지 아니하다.\n② 중앙행정기관의 장이 지역ㆍ지구등을 지정하는 경우에는 지적(地籍)이 표시된 지형도에 지역ㆍ지구등을 명시한 도면(이하 ""지형도면""이라 한다)을 작성하여 관보에 고시하고, 지방자치단체의 장이 지역ㆍ지구등을 지정하는 경우에는 지형도면을 작성하여 그 지방자치단체의 공보에 고시하여야 한다. 다만, 대통령령으로 정하는 경우에는 지형도면을 작성ㆍ고시하지 아니하거나 지적도 등에 지역ㆍ지구등을 명시한 도면을 작성하여 고시할 수 있다.\n③ 제2항에 따라 지형도면 또는 지적도 등에 지역ㆍ지구등을 명시한 도면(이하 ""지형도면등""이라 한다)을 고시하여야 하는 지역ㆍ지구등의 지정의 효력은 지형도면등의 고시를 함으로써 발생한다. 다만, 지역ㆍ지구등을 지정할 때에 지형도면등의 고시가 곤란한 경우로서 대통령령으로 정하는 경우에는 그러하지 아니하다.\n④ 제3항 단서에 해당되는 경우에는 지역ㆍ지구등의 지정일부터 2년이 되는 날까지 지형도면등을 고시하여야 하며, 지형도면등의 고시가 없는 경우에는 그 2년이 되는 날의 다음 날부터 그 지정의 효력을 잃는다.\n⑤ 제4항에 따라 지역ㆍ지구등의 지정이 효력을 잃은 때에는 그 지역ㆍ지구등의 지정권자는 대통령령으로 정하는 바에 따라 지체 없이 그 사실을 관보 또는 공보에 고시하고, 이를 관계 특별자치도지사ㆍ시장ㆍ군수(광역시의 관할 구역에 있는 군의 군수를 포함한다. 이하 같다) 또는 구청장(구청장은 자치구의 구청장을 말하며, 이하 ""시장ㆍ군수 또는 구청장""이라 한다)에게 통보하여야 한다. 이 경우 시장ㆍ군수 또는 구청장은 그 내용을 제12조에 따른 국토이용정보체계(이하 ""국토이용정보체계""라 한다)에 등재(登載)하여 일반 국민이 볼 수 있도록 하여야 한다.\n⑥ 중앙행정기관의 장이나 지방자치단체의 장은 지역ㆍ지구등의 지정을 입안하거나 신청하는 자가 따로 있는 경우에는 그 자에게 제2항에 따른 고시에 필요한 지형도면등을 작성하여 제출하도록 요청할 수 있다.\n⑦ 제2항에 따른 지형도면등의 작성에 필요한 구체적인 기준 및 방법 등은 대통령령으로 정한다.\n⑧ 중앙행정기관의 장이나 지방자치단체의 장은 제2항에 따라 지형도면등의 고시를 하려면 관계 시장ㆍ군수 또는 구청장에게 관련 서류와 고시예정일 등 대통령령으로 정하는 사항을 미리 통보하여야 한다. 다만, 제2항 단서에 따라 지형도면을 작성ㆍ고시하지 아니하는 경우에는 지역ㆍ지구등을 지정할 때에 대통령령으로 정하는 사항을 미리 통보하여야 하고, 제3항 단서에 따라 지역ㆍ지구등의 지정 후에 지형도면등의 고시를 하는 경우에는 지역ㆍ지구등을 지정할 때와 제4항에 따른 지형도면등을 고시할 때에 대통령령으로 정하는 사항을 미리 통보하여야 한다.\n⑨ 제8항에 따라 통보를 받은 시장ㆍ군수 또는 구청장은 그 내용을 국토이용정보체계에 등재하여 지역ㆍ지구등의 지정 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. 다만, 제3항 단서에 따라 지역ㆍ지구등의 지정 후에 지형도면등의 고시를 하는 경우에는 제4항에 따라 지형도면등을 고시한 날부터 일반 국민이 볼 수 있도록 하여야 한다.\n제9조(지역ㆍ지구등의 지정 및 행위제한 내용의 제공)\n① 국토교통부장관과 지방자치단체의 장은 국토이용정보체계를 이용하여 필지별로 지역ㆍ지구등의 지정 여부 및 행위제한 내용을 일반 국민에게 제공하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장은 지역ㆍ지구등이 신설되거나 지역ㆍ지구등에서의 행위제한 내용이 변경되는 경우에는 그 내용을 대통령령으로 정하는 바에 따라 국토교통부장관에게 통보하여야 한다. 이 경우 국토교통부장관은 국토이용정보체계를 통하여 제공되는 내용을 변경하여야 한다. <개정 2013.3.23>\n③ 지방자치단체의 장은 지역ㆍ지구등이 신설되거나 지역ㆍ지구등에서의 행위제한 내용이 변경되는 경우에는 그 내용을 대통령령으로 정하는 바에 따라 국토교통부장관에게 통보하고 국토이용정보체계를 통하여 제공되는 내용을 직접 변경하여야 한다. <개정 2013.3.23>\n제10조(토지이용계획확인서의 발급 등)\n① 시장ㆍ군수 또는 구청장은 다음 각 호의 사항을 확인하는 서류(이하 ""토지이용계획확인서""라 한다)의 발급 신청이 있는 경우에는 대통령령으로 정하는 바에 따라 토지이용계획확인서를 발급하여야 한다.\n② 제1항에 따라 토지이용계획확인서의 발급을 신청하는 자는 시장ㆍ군수 또는 구청장에게 그 지방자치단체의 조례로 정하는 수수료를 내야 한다.\n제11조(규제안내서)\n① 국토교통부장관은 규제안내서를 작성할 수 있다. <개정 2013.3.23>\n② 국토교통부장관이 규제안내서를 작성하려면 관계 행정기관의 장과 미리 협의하여야 한다. 이 경우 협의를 요청받은 관계 행정기관의 장은 특별한 사유가 없으면 그 요청을 받은 날부터 30일 이내에 의견을 제시하여야 한다. <개정 2013.3.23>\n③ 국토교통부장관이 규제안내서를 작성한 경우에는 이를 관보에 고시하여야 하며, 국토이용정보체계를 이용하여 일반 국민에게 제공하여야 한다. <개정 2013.3.23>\n④ 규제안내서에는 다음 각 호의 사항이 포함되어야 한다.\n⑤ 중앙행정기관의 장이 제3항에 따라 고시된 규제안내서에 포함된 내용을 변경하는 경우에는 그 내용을 변경하는 법령의 공포일에 규제안내서의 내용이 변경된 사실과 그 효력 발생일을 함께 관보에 고시하여야 하며, 고시를 하기 전에 미리 고시예정일 등 대통령령으로 정하는 사항을 국토교통부장관에게 통보하여야 한다. 이 경우 국토교통부장관은 국토이용정보체계를 통하여 제공되는 규제안내서를 변경하여 그 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. <개정 2013.3.23>\n⑥ 지방자치단체의 장이 제3항에 따라 고시된 규제안내서에 포함된 내용을 변경하는 경우에는 그 내용을 변경하는 자치법규의 공포일에 규제안내서의 내용이 변경된 사실과 그 효력 발생일을 함께 공보에 고시하여야 하며, 고시를 하기 전에 미리 고시예정일 등 대통령령으로 정하는 사항을 국토교통부장관에게 통보하여야 한다. 이 경우 지방자치단체의 장은 국토이용정보체계를 통하여 제공되는 규제안내서를 변경하여 그 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. <개정 2013.3.23>\n제12조(국토이용정보체계의 구축ㆍ운영 및 활용)\n① 국토교통부장관, 특별시장, 광역시장, 도지사, 시장ㆍ군수 또는 구청장(이하 ""정보체계운영자""라 한다)은 국토의 이용 및 관리 업무를 효율적으로 추진하기 위하여 국토이용정보체계를 구축하여 운영할 수 있다. <개정 2013.3.23>\n② 정보체계운영자는 국토이용정보체계를 통하여 다음 각 호의 사항을 일반 국민에게 제공할 수 있다.\n③ 정보체계운영자는 국토이용정보체계를 효율적으로 만들어 운영하거나 활용하기 위하여 필요하면 전담부서를 설치할 수 있다.\n④ 행정안전부장관 등 관계 행정기관의 장은 제3항에 따라 정보체계운영자가 전담부서를 설치하려는 경우에는 이에 협조하여야 한다. <개정 2013.3.23, 2014.11.19, 2017.7.26>\n⑤ 국토이용정보체계를 통하여 관리되는 정보의 내용과 국토이용정보체계의 구축ㆍ운영 또는 이를 활용한 정보의 제공 및 그 업무 처리에 필요한 사항은 대통령령으로 정한다.\n제13조(지역ㆍ지구등의 지정과 운영 실적 등의 평가)\n① 지역ㆍ지구등을 관장하는 중앙행정기관의 장 및 지방자치단체의 장은 2년마다 지역ㆍ지구등의 지정과 운영 실적 등을 포함한 토지이용규제보고서를 작성하여 국토교통부장관에게 제출하여야 한다. <개정 2013.3.23>\n② 국토교통부장관은 토지이용규제의 적정성을 확보하기 위하여 제22조에 따라 설치된 토지이용규제평가단(이하 ""평가단""이라 한다)으로 하여금 제1항에 따라 제출된 토지이용규제보고서에 기초하여 지역ㆍ지구등의 지정 실태 등을 평가하게 하고, 위원회의 심의를 거쳐 국무회의에 보고한 후 중앙행정기관의 장 또는 지방자치단체의 장에게 그 지역ㆍ지구등의 통합이나 폐합 등 제도개선을 요청할 수 있다. <개정 2013.3.23>\n③ 제2항에 따라 제도개선을 요청받은 중앙행정기관의 장 또는 지방자치단체의 장은 특별한 사유가 없으면 지역ㆍ지구등의 통합이나 폐합 등을 위한 법령 또는 자치법규의 개정 방안, 지역ㆍ지구등을 대체할 수 있는 제도의 신설 등의 대책을 마련하여 국토교통부장관과 협의하여야 한다. <개정 2013.3.23>\n④ 토지이용규제보고서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제14조(행위제한 내용 및 절차에 대한 평가) 국토교통부장관은 서로 다른 지역ㆍ지구등에서 행위제한 내용 및 절차의 균형이 유지되도록 하기 위하여 매년 대통령령으로 정하는 바에 따라 평가단으로 하여금 지역ㆍ지구등에서의 행위제한 내용 및 절차를 조사하여 평가하게 하고, 평가 결과에 대하여 위원회의 심의를 거쳐 중앙행정기관의 장이나 지방자치단체의 장에게 제도개선을 요청할 수 있다. <개정 2013.3.23>\n제15조(토지이용규제심의위원회)\n① 지역ㆍ지구등의 신설 등에 관한 사항을 심의하기 위하여 국토교통부에 토지이용규제심의위원회를 둔다. <개정 2013.3.23>\n② 위원회는 다음 각 호의 사항을 심의한다.\n제16조(위원회의 구성 등)\n① 위원회는 위원장과 부위원장 각 1명을 포함한 20명 이내의 위원으로 구성한다.\n② 위원회의 위원장은 국토교통부장관이 되고, 부위원장은 환경부차관이 된다. <개정 2013.3.23>\n③ 위원장과 부위원장을 제외한 위원은 다음 각 호의 사람이 된다. <개정 2013.3.23>\n④ 위촉위원의 임기는 2년으로 한다.\n제17조(위원의 결격사유)\n① 다음 각 호의 어느 하나에 해당하는 사람은 위원회의 위원이 될 수 없다. <개정 2017.4.18>\n② 위원이 제1항 각 호의 어느 하나에 해당하게 된 때에는 그 날로 위원자격을 잃는다.\n제18조(위원장 등의 직무)\n① 위원회의 위원장은 위원회를 대표하고, 위원회의 업무를 총괄한다.\n② 위원회의 부위원장은 위원장을 보좌하며, 위원장이 부득이한 사유로 직무를 수행할 수 없을 때에는 그 직무를 대행한다.\n③ 위원장과 부위원장이 모두 부득이한 사유로 직무를 수행할 수 없을 때에는 위원장이 미리 지명한 위원이 그 직무를 대행한다.\n제19조(회의의 소집 및 의결정족수)\n① 위원회의 위원장은 위원회의 회의를 소집하고, 그 의장이 된다.\n② 위원회의 회의는 재적위원 과반수의 출석으로 개의(開議)하고, 출석위원 과반수의 찬성으로 의결한다. 다만, 제15조제2항제2호에서 규정한 사항은 재적위원 과반수의 찬성으로 의결한다.\n제20조(간사 및 서기)\n① 위원회에 간사와 서기를 둔다.\n② 간사와 서기는 국토교통부 소속 공무원 중에서 위원장이 임명한다. <개정 2013.3.23>\n③ 간사는 위원장의 명을 받아 위원회의 사무를 담당하고, 서기는 간사를 보좌한다.\n제21조(운영세칙) 위원회의 설치 및 운영에 필요한 사항은 대통령령으로 정한다.\n제22조(토지이용규제평가단)\n① 다음 각 호의 업무를 처리하기 위하여 위원회에 토지이용규제평가단을 설치하여 운영할 수 있다.\n② 평가단의 단장은 위촉위원들이 위촉위원 중에서 단장으로 뽑은 사람이 된다.\n③ 평가단의 구성 및 운영에 필요한 사항은 대통령령으로 정한다.\n제23조(업무의 위탁) 정보체계운영자는 국토이용정보체계의 운영을 대통령령으로 정하는 기관 또는 단체에 위탁할 수 있다.\n제24조(벌칙 적용 시의 공무원 의제) 다음 각 호의 어느 하나에 해당하는 자는 「형법」 제127조 및 제129조부터 제132조까지의 규정을 적용할 때에는 공무원으로 본다.\n\n', 'avg_bleu_4': 0.002763049099007199, 'avg_rouge_1': 0.02844969186838143, 'avg_rouge_l': 0.02844969186838143, 'avg_gpt_score': 1.375}
    {'label': '토지이용규제 기본법', 'full_text': '토지이용규제 기본법\n[시행20170726] [법률 제14839호, 20170726, 타법개정]\n제1조(목적) 이 법은 토지이용과 관련된 지역ㆍ지구등의 지정과 관리에 관한 기본적인 사항을 규정함으로써 토지이용규제의 투명성을 확보하여 국민의 토지이용상의 불편을 줄이고 국민경제의 발전에 이바지함을 목적으로 한다.\n제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다. <개정 2011.4.14>\n\n제3조(다른 법률과의 관계) 지역ㆍ지구등의 지정(따로 지정 절차 없이 법령 또는 자치법규에 따라 지역ㆍ지구등의 범위가 직접 지정되는 경우를 포함한다. 이하 같다)과 운영 등에 관하여 다른 법률에 제8조와 다른 규정이 있는 경우에는 이 법에 따른다.\n제4조(토지이용규제의 투명성 확보) 지역ㆍ지구등을 규정하는 법령 또는 자치법규는 그 지정목적, 지정기준, 행위제한내용 등을 구체적이고 명확하게 규정하여야 한다.\n제5조(지역ㆍ지구등의 신설 제한 등) 지역ㆍ지구등은 다음 각 호에 규정된 것 외에는 신설(지역ㆍ지구등을 세분하거나 변경하는 것을 포함한다. 이하 같다)할 수 없다. <개정 2013.3.23>\n\n제6조(지역ㆍ지구등의 신설에 대한 심의)\n① 중앙행정기관의 장이나 지방자치단체의 장은 지역ㆍ지구등을 신설하는 내용으로 법령 또는 자치법규를 제정하거나 개정하려면 해당 법령안 또는 자치법규안을 입법예고하기 전에 신설될 지역ㆍ지구등이 다음 각 호의 기준에 부합하는지에 대하여 제15조에 따른 토지이용규제심의위원회(이하 ""위원회""라 한다)의 심의를 국토교통부장관에게 요청하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장이나 지방자치단체의 장은 제1항에 따른 심의를 요청할 때에는 지역ㆍ지구등의 지정 및 운영계획서(이하 이 조에서 ""운영계획서""라 한다)를 작성하여 제출하여야 한다.\n③ 국토교통부장관은 제1항에 따른 심의 결과 지역ㆍ지구등의 신설이 제1항 각 호의 기준에 부합하지 아니한다고 인정하는 경우에는 운영계획서를 제출한 중앙행정기관의 장이나 지방자치단체의 장에게 운영계획서의 재검토 또는 수정을 요청할 수 있다. <개정 2013.3.23>\n④ 운영계획서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제6조의2(행위제한 강화등에 대한 심의)\n① 중앙행정기관의 장이나 지방자치단체의 장은 제5조 각 호의 지역ㆍ지구등에서의 행위제한을 신설 또는 강화(이하 ""강화등""이라 한다)하려는 경우에는 해당 법령안 또는 자치법규안을 입법예고하기 전에 다음 각 호의 기준에 부합하는지에 대하여 위원회의 심의를 국토교통부장관에게 요청하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장이나 지방자치단체의 장은 제1항에 따라 심의를 요청할 때에는 행위제한 강화등 계획서(이하 이 조에서 ""계획서""라 한다)를 작성하여 제출하여야 한다.\n③ 국토교통부장관은 제1항에 따른 심의결과 행위제한 강화등이 제1항 각 호의 기준에 부합하지 아니한다고 인정하는 경우에는 계획서를 제출한 중앙행정기관의 장이나 지방자치단체의 장에게 계획서의 재검토 또는 수정을 요청할 수 있다. <개정 2013.3.23>\n④ 계획서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제7조(사업지구에서의 행위제한 등)\n① 개발사업을 시행하기 위한 지역ㆍ지구등(이하 이 조에서 ""사업지구""라 한다)을 규정하는 법령 또는 자치법규는 해당 사업지구에서 개발사업에 지장을 초래할 수 있는 다음 각 호의 행위로서 관계 행정기관의 장의 허가 또는 변경허가를 받아야 하는 사항을 구체적으로 정하여야 한다.\n② 사업지구를 규정하는 법령 또는 자치법규는 다음 각 호의 사항을 구체적으로 정하여야 한다.\n제8조(지역ㆍ지구등의 지정 등)\n① 중앙행정기관의 장이나 지방자치단체의 장이 지역ㆍ지구등을 지정(변경을 포함한다. 이하 같다)하려면 대통령령으로 정하는 바에 따라 미리 주민의 의견을 들어야 한다. 다만, 다음 각 호의 어느 하나에 해당하거나 대통령령으로 정하는 경미한 사항을 변경하는 경우에는 그러하지 아니하다.\n② 중앙행정기관의 장이 지역ㆍ지구등을 지정하는 경우에는 지적(地籍)이 표시된 지형도에 지역ㆍ지구등을 명시한 도면(이하 ""지형도면""이라 한다)을 작성하여 관보에 고시하고, 지방자치단체의 장이 지역ㆍ지구등을 지정하는 경우에는 지형도면을 작성하여 그 지방자치단체의 공보에 고시하여야 한다. 다만, 대통령령으로 정하는 경우에는 지형도면을 작성ㆍ고시하지 아니하거나 지적도 등에 지역ㆍ지구등을 명시한 도면을 작성하여 고시할 수 있다.\n③ 제2항에 따라 지형도면 또는 지적도 등에 지역ㆍ지구등을 명시한 도면(이하 ""지형도면등""이라 한다)을 고시하여야 하는 지역ㆍ지구등의 지정의 효력은 지형도면등의 고시를 함으로써 발생한다. 다만, 지역ㆍ지구등을 지정할 때에 지형도면등의 고시가 곤란한 경우로서 대통령령으로 정하는 경우에는 그러하지 아니하다.\n④ 제3항 단서에 해당되는 경우에는 지역ㆍ지구등의 지정일부터 2년이 되는 날까지 지형도면등을 고시하여야 하며, 지형도면등의 고시가 없는 경우에는 그 2년이 되는 날의 다음 날부터 그 지정의 효력을 잃는다.\n⑤ 제4항에 따라 지역ㆍ지구등의 지정이 효력을 잃은 때에는 그 지역ㆍ지구등의 지정권자는 대통령령으로 정하는 바에 따라 지체 없이 그 사실을 관보 또는 공보에 고시하고, 이를 관계 특별자치도지사ㆍ시장ㆍ군수(광역시의 관할 구역에 있는 군의 군수를 포함한다. 이하 같다) 또는 구청장(구청장은 자치구의 구청장을 말하며, 이하 ""시장ㆍ군수 또는 구청장""이라 한다)에게 통보하여야 한다. 이 경우 시장ㆍ군수 또는 구청장은 그 내용을 제12조에 따른 국토이용정보체계(이하 ""국토이용정보체계""라 한다)에 등재(登載)하여 일반 국민이 볼 수 있도록 하여야 한다.\n⑥ 중앙행정기관의 장이나 지방자치단체의 장은 지역ㆍ지구등의 지정을 입안하거나 신청하는 자가 따로 있는 경우에는 그 자에게 제2항에 따른 고시에 필요한 지형도면등을 작성하여 제출하도록 요청할 수 있다.\n⑦ 제2항에 따른 지형도면등의 작성에 필요한 구체적인 기준 및 방법 등은 대통령령으로 정한다.\n⑧ 중앙행정기관의 장이나 지방자치단체의 장은 제2항에 따라 지형도면등의 고시를 하려면 관계 시장ㆍ군수 또는 구청장에게 관련 서류와 고시예정일 등 대통령령으로 정하는 사항을 미리 통보하여야 한다. 다만, 제2항 단서에 따라 지형도면을 작성ㆍ고시하지 아니하는 경우에는 지역ㆍ지구등을 지정할 때에 대통령령으로 정하는 사항을 미리 통보하여야 하고, 제3항 단서에 따라 지역ㆍ지구등의 지정 후에 지형도면등의 고시를 하는 경우에는 지역ㆍ지구등을 지정할 때와 제4항에 따른 지형도면등을 고시할 때에 대통령령으로 정하는 사항을 미리 통보하여야 한다.\n⑨ 제8항에 따라 통보를 받은 시장ㆍ군수 또는 구청장은 그 내용을 국토이용정보체계에 등재하여 지역ㆍ지구등의 지정 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. 다만, 제3항 단서에 따라 지역ㆍ지구등의 지정 후에 지형도면등의 고시를 하는 경우에는 제4항에 따라 지형도면등을 고시한 날부터 일반 국민이 볼 수 있도록 하여야 한다.\n제9조(지역ㆍ지구등의 지정 및 행위제한 내용의 제공)\n① 국토교통부장관과 지방자치단체의 장은 국토이용정보체계를 이용하여 필지별로 지역ㆍ지구등의 지정 여부 및 행위제한 내용을 일반 국민에게 제공하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장은 지역ㆍ지구등이 신설되거나 지역ㆍ지구등에서의 행위제한 내용이 변경되는 경우에는 그 내용을 대통령령으로 정하는 바에 따라 국토교통부장관에게 통보하여야 한다. 이 경우 국토교통부장관은 국토이용정보체계를 통하여 제공되는 내용을 변경하여야 한다. <개정 2013.3.23>\n③ 지방자치단체의 장은 지역ㆍ지구등이 신설되거나 지역ㆍ지구등에서의 행위제한 내용이 변경되는 경우에는 그 내용을 대통령령으로 정하는 바에 따라 국토교통부장관에게 통보하고 국토이용정보체계를 통하여 제공되는 내용을 직접 변경하여야 한다. <개정 2013.3.23>\n제10조(토지이용계획확인서의 발급 등)\n① 시장ㆍ군수 또는 구청장은 다음 각 호의 사항을 확인하는 서류(이하 ""토지이용계획확인서""라 한다)의 발급 신청이 있는 경우에는 대통령령으로 정하는 바에 따라 토지이용계획확인서를 발급하여야 한다.\n② 제1항에 따라 토지이용계획확인서의 발급을 신청하는 자는 시장ㆍ군수 또는 구청장에게 그 지방자치단체의 조례로 정하는 수수료를 내야 한다.\n제11조(규제안내서)\n① 국토교통부장관은 규제안내서를 작성할 수 있다. <개정 2013.3.23>\n② 국토교통부장관이 규제안내서를 작성하려면 관계 행정기관의 장과 미리 협의하여야 한다. 이 경우 협의를 요청받은 관계 행정기관의 장은 특별한 사유가 없으면 그 요청을 받은 날부터 30일 이내에 의견을 제시하여야 한다. <개정 2013.3.23>\n③ 국토교통부장관이 규제안내서를 작성한 경우에는 이를 관보에 고시하여야 하며, 국토이용정보체계를 이용하여 일반 국민에게 제공하여야 한다. <개정 2013.3.23>\n④ 규제안내서에는 다음 각 호의 사항이 포함되어야 한다.\n⑤ 중앙행정기관의 장이 제3항에 따라 고시된 규제안내서에 포함된 내용을 변경하는 경우에는 그 내용을 변경하는 법령의 공포일에 규제안내서의 내용이 변경된 사실과 그 효력 발생일을 함께 관보에 고시하여야 하며, 고시를 하기 전에 미리 고시예정일 등 대통령령으로 정하는 사항을 국토교통부장관에게 통보하여야 한다. 이 경우 국토교통부장관은 국토이용정보체계를 통하여 제공되는 규제안내서를 변경하여 그 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. <개정 2013.3.23>\n⑥ 지방자치단체의 장이 제3항에 따라 고시된 규제안내서에 포함된 내용을 변경하는 경우에는 그 내용을 변경하는 자치법규의 공포일에 규제안내서의 내용이 변경된 사실과 그 효력 발생일을 함께 공보에 고시하여야 하며, 고시를 하기 전에 미리 고시예정일 등 대통령령으로 정하는 사항을 국토교통부장관에게 통보하여야 한다. 이 경우 지방자치단체의 장은 국토이용정보체계를 통하여 제공되는 규제안내서를 변경하여 그 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. <개정 2013.3.23>\n제12조(국토이용정보체계의 구축ㆍ운영 및 활용)\n① 국토교통부장관, 특별시장, 광역시장, 도지사, 시장ㆍ군수 또는 구청장(이하 ""정보체계운영자""라 한다)은 국토의 이용 및 관리 업무를 효율적으로 추진하기 위하여 국토이용정보체계를 구축하여 운영할 수 있다. <개정 2013.3.23>\n② 정보체계운영자는 국토이용정보체계를 통하여 다음 각 호의 사항을 일반 국민에게 제공할 수 있다.\n③ 정보체계운영자는 국토이용정보체계를 효율적으로 만들어 운영하거나 활용하기 위하여 필요하면 전담부서를 설치할 수 있다.\n④ 행정안전부장관 등 관계 행정기관의 장은 제3항에 따라 정보체계운영자가 전담부서를 설치하려는 경우에는 이에 협조하여야 한다. <개정 2013.3.23, 2014.11.19, 2017.7.26>\n⑤ 국토이용정보체계를 통하여 관리되는 정보의 내용과 국토이용정보체계의 구축ㆍ운영 또는 이를 활용한 정보의 제공 및 그 업무 처리에 필요한 사항은 대통령령으로 정한다.\n제13조(지역ㆍ지구등의 지정과 운영 실적 등의 평가)\n① 지역ㆍ지구등을 관장하는 중앙행정기관의 장 및 지방자치단체의 장은 2년마다 지역ㆍ지구등의 지정과 운영 실적 등을 포함한 토지이용규제보고서를 작성하여 국토교통부장관에게 제출하여야 한다. <개정 2013.3.23>\n② 국토교통부장관은 토지이용규제의 적정성을 확보하기 위하여 제22조에 따라 설치된 토지이용규제평가단(이하 ""평가단""이라 한다)으로 하여금 제1항에 따라 제출된 토지이용규제보고서에 기초하여 지역ㆍ지구등의 지정 실태 등을 평가하게 하고, 위원회의 심의를 거쳐 국무회의에 보고한 후 중앙행정기관의 장 또는 지방자치단체의 장에게 그 지역ㆍ지구등의 통합이나 폐합 등 제도개선을 요청할 수 있다. <개정 2013.3.23>\n③ 제2항에 따라 제도개선을 요청받은 중앙행정기관의 장 또는 지방자치단체의 장은 특별한 사유가 없으면 지역ㆍ지구등의 통합이나 폐합 등을 위한 법령 또는 자치법규의 개정 방안, 지역ㆍ지구등을 대체할 수 있는 제도의 신설 등의 대책을 마련하여 국토교통부장관과 협의하여야 한다. <개정 2013.3.23>\n④ 토지이용규제보고서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제14조(행위제한 내용 및 절차에 대한 평가) 국토교통부장관은 서로 다른 지역ㆍ지구등에서 행위제한 내용 및 절차의 균형이 유지되도록 하기 위하여 매년 대통령령으로 정하는 바에 따라 평가단으로 하여금 지역ㆍ지구등에서의 행위제한 내용 및 절차를 조사하여 평가하게 하고, 평가 결과에 대하여 위원회의 심의를 거쳐 중앙행정기관의 장이나 지방자치단체의 장에게 제도개선을 요청할 수 있다. <개정 2013.3.23>\n제15조(토지이용규제심의위원회)\n① 지역ㆍ지구등의 신설 등에 관한 사항을 심의하기 위하여 국토교통부에 토지이용규제심의위원회를 둔다. <개정 2013.3.23>\n② 위원회는 다음 각 호의 사항을 심의한다.\n제16조(위원회의 구성 등)\n① 위원회는 위원장과 부위원장 각 1명을 포함한 20명 이내의 위원으로 구성한다.\n② 위원회의 위원장은 국토교통부장관이 되고, 부위원장은 환경부차관이 된다. <개정 2013.3.23>\n③ 위원장과 부위원장을 제외한 위원은 다음 각 호의 사람이 된다. <개정 2013.3.23>\n④ 위촉위원의 임기는 2년으로 한다.\n제17조(위원의 결격사유)\n① 다음 각 호의 어느 하나에 해당하는 사람은 위원회의 위원이 될 수 없다. <개정 2017.4.18>\n② 위원이 제1항 각 호의 어느 하나에 해당하게 된 때에는 그 날로 위원자격을 잃는다.\n제18조(위원장 등의 직무)\n① 위원회의 위원장은 위원회를 대표하고, 위원회의 업무를 총괄한다.\n② 위원회의 부위원장은 위원장을 보좌하며, 위원장이 부득이한 사유로 직무를 수행할 수 없을 때에는 그 직무를 대행한다.\n③ 위원장과 부위원장이 모두 부득이한 사유로 직무를 수행할 수 없을 때에는 위원장이 미리 지명한 위원이 그 직무를 대행한다.\n제19조(회의의 소집 및 의결정족수)\n① 위원회의 위원장은 위원회의 회의를 소집하고, 그 의장이 된다.\n② 위원회의 회의는 재적위원 과반수의 출석으로 개의(開議)하고, 출석위원 과반수의 찬성으로 의결한다. 다만, 제15조제2항제2호에서 규정한 사항은 재적위원 과반수의 찬성으로 의결한다.\n제20조(간사 및 서기)\n① 위원회에 간사와 서기를 둔다.\n② 간사와 서기는 국토교통부 소속 공무원 중에서 위원장이 임명한다. <개정 2013.3.23>\n③ 간사는 위원장의 명을 받아 위원회의 사무를 담당하고, 서기는 간사를 보좌한다.\n제21조(운영세칙) 위원회의 설치 및 운영에 필요한 사항은 대통령령으로 정한다.\n제22조(토지이용규제평가단)\n① 다음 각 호의 업무를 처리하기 위하여 위원회에 토지이용규제평가단을 설치하여 운영할 수 있다.\n② 평가단의 단장은 위촉위원들이 위촉위원 중에서 단장으로 뽑은 사람이 된다.\n③ 평가단의 구성 및 운영에 필요한 사항은 대통령령으로 정한다.\n제23조(업무의 위탁) 정보체계운영자는 국토이용정보체계의 운영을 대통령령으로 정하는 기관 또는 단체에 위탁할 수 있다.\n제24조(벌칙 적용 시의 공무원 의제) 다음 각 호의 어느 하나에 해당하는 자는 「형법」 제127조 및 제129조부터 제132조까지의 규정을 적용할 때에는 공무원으로 본다.\n\n', 'avg_bleu_4': 0.002763049099007199, 'avg_rouge_1': 0.02844969186838143, 'avg_rouge_l': 0.02844969186838143, 'avg_gpt_score': 1.375}
    {'label': '토지이용규제 기본법', 'full_text': '토지이용규제 기본법\n[시행20170726] [법률 제14839호, 20170726, 타법개정]\n제1조(목적) 이 법은 토지이용과 관련된 지역ㆍ지구등의 지정과 관리에 관한 기본적인 사항을 규정함으로써 토지이용규제의 투명성을 확보하여 국민의 토지이용상의 불편을 줄이고 국민경제의 발전에 이바지함을 목적으로 한다.\n제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다. <개정 2011.4.14>\n\n제3조(다른 법률과의 관계) 지역ㆍ지구등의 지정(따로 지정 절차 없이 법령 또는 자치법규에 따라 지역ㆍ지구등의 범위가 직접 지정되는 경우를 포함한다. 이하 같다)과 운영 등에 관하여 다른 법률에 제8조와 다른 규정이 있는 경우에는 이 법에 따른다.\n제4조(토지이용규제의 투명성 확보) 지역ㆍ지구등을 규정하는 법령 또는 자치법규는 그 지정목적, 지정기준, 행위제한내용 등을 구체적이고 명확하게 규정하여야 한다.\n제5조(지역ㆍ지구등의 신설 제한 등) 지역ㆍ지구등은 다음 각 호에 규정된 것 외에는 신설(지역ㆍ지구등을 세분하거나 변경하는 것을 포함한다. 이하 같다)할 수 없다. <개정 2013.3.23>\n\n제6조(지역ㆍ지구등의 신설에 대한 심의)\n① 중앙행정기관의 장이나 지방자치단체의 장은 지역ㆍ지구등을 신설하는 내용으로 법령 또는 자치법규를 제정하거나 개정하려면 해당 법령안 또는 자치법규안을 입법예고하기 전에 신설될 지역ㆍ지구등이 다음 각 호의 기준에 부합하는지에 대하여 제15조에 따른 토지이용규제심의위원회(이하 ""위원회""라 한다)의 심의를 국토교통부장관에게 요청하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장이나 지방자치단체의 장은 제1항에 따른 심의를 요청할 때에는 지역ㆍ지구등의 지정 및 운영계획서(이하 이 조에서 ""운영계획서""라 한다)를 작성하여 제출하여야 한다.\n③ 국토교통부장관은 제1항에 따른 심의 결과 지역ㆍ지구등의 신설이 제1항 각 호의 기준에 부합하지 아니한다고 인정하는 경우에는 운영계획서를 제출한 중앙행정기관의 장이나 지방자치단체의 장에게 운영계획서의 재검토 또는 수정을 요청할 수 있다. <개정 2013.3.23>\n④ 운영계획서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제6조의2(행위제한 강화등에 대한 심의)\n① 중앙행정기관의 장이나 지방자치단체의 장은 제5조 각 호의 지역ㆍ지구등에서의 행위제한을 신설 또는 강화(이하 ""강화등""이라 한다)하려는 경우에는 해당 법령안 또는 자치법규안을 입법예고하기 전에 다음 각 호의 기준에 부합하는지에 대하여 위원회의 심의를 국토교통부장관에게 요청하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장이나 지방자치단체의 장은 제1항에 따라 심의를 요청할 때에는 행위제한 강화등 계획서(이하 이 조에서 ""계획서""라 한다)를 작성하여 제출하여야 한다.\n③ 국토교통부장관은 제1항에 따른 심의결과 행위제한 강화등이 제1항 각 호의 기준에 부합하지 아니한다고 인정하는 경우에는 계획서를 제출한 중앙행정기관의 장이나 지방자치단체의 장에게 계획서의 재검토 또는 수정을 요청할 수 있다. <개정 2013.3.23>\n④ 계획서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제7조(사업지구에서의 행위제한 등)\n① 개발사업을 시행하기 위한 지역ㆍ지구등(이하 이 조에서 ""사업지구""라 한다)을 규정하는 법령 또는 자치법규는 해당 사업지구에서 개발사업에 지장을 초래할 수 있는 다음 각 호의 행위로서 관계 행정기관의 장의 허가 또는 변경허가를 받아야 하는 사항을 구체적으로 정하여야 한다.\n② 사업지구를 규정하는 법령 또는 자치법규는 다음 각 호의 사항을 구체적으로 정하여야 한다.\n제8조(지역ㆍ지구등의 지정 등)\n① 중앙행정기관의 장이나 지방자치단체의 장이 지역ㆍ지구등을 지정(변경을 포함한다. 이하 같다)하려면 대통령령으로 정하는 바에 따라 미리 주민의 의견을 들어야 한다. 다만, 다음 각 호의 어느 하나에 해당하거나 대통령령으로 정하는 경미한 사항을 변경하는 경우에는 그러하지 아니하다.\n② 중앙행정기관의 장이 지역ㆍ지구등을 지정하는 경우에는 지적(地籍)이 표시된 지형도에 지역ㆍ지구등을 명시한 도면(이하 ""지형도면""이라 한다)을 작성하여 관보에 고시하고, 지방자치단체의 장이 지역ㆍ지구등을 지정하는 경우에는 지형도면을 작성하여 그 지방자치단체의 공보에 고시하여야 한다. 다만, 대통령령으로 정하는 경우에는 지형도면을 작성ㆍ고시하지 아니하거나 지적도 등에 지역ㆍ지구등을 명시한 도면을 작성하여 고시할 수 있다.\n③ 제2항에 따라 지형도면 또는 지적도 등에 지역ㆍ지구등을 명시한 도면(이하 ""지형도면등""이라 한다)을 고시하여야 하는 지역ㆍ지구등의 지정의 효력은 지형도면등의 고시를 함으로써 발생한다. 다만, 지역ㆍ지구등을 지정할 때에 지형도면등의 고시가 곤란한 경우로서 대통령령으로 정하는 경우에는 그러하지 아니하다.\n④ 제3항 단서에 해당되는 경우에는 지역ㆍ지구등의 지정일부터 2년이 되는 날까지 지형도면등을 고시하여야 하며, 지형도면등의 고시가 없는 경우에는 그 2년이 되는 날의 다음 날부터 그 지정의 효력을 잃는다.\n⑤ 제4항에 따라 지역ㆍ지구등의 지정이 효력을 잃은 때에는 그 지역ㆍ지구등의 지정권자는 대통령령으로 정하는 바에 따라 지체 없이 그 사실을 관보 또는 공보에 고시하고, 이를 관계 특별자치도지사ㆍ시장ㆍ군수(광역시의 관할 구역에 있는 군의 군수를 포함한다. 이하 같다) 또는 구청장(구청장은 자치구의 구청장을 말하며, 이하 ""시장ㆍ군수 또는 구청장""이라 한다)에게 통보하여야 한다. 이 경우 시장ㆍ군수 또는 구청장은 그 내용을 제12조에 따른 국토이용정보체계(이하 ""국토이용정보체계""라 한다)에 등재(登載)하여 일반 국민이 볼 수 있도록 하여야 한다.\n⑥ 중앙행정기관의 장이나 지방자치단체의 장은 지역ㆍ지구등의 지정을 입안하거나 신청하는 자가 따로 있는 경우에는 그 자에게 제2항에 따른 고시에 필요한 지형도면등을 작성하여 제출하도록 요청할 수 있다.\n⑦ 제2항에 따른 지형도면등의 작성에 필요한 구체적인 기준 및 방법 등은 대통령령으로 정한다.\n⑧ 중앙행정기관의 장이나 지방자치단체의 장은 제2항에 따라 지형도면등의 고시를 하려면 관계 시장ㆍ군수 또는 구청장에게 관련 서류와 고시예정일 등 대통령령으로 정하는 사항을 미리 통보하여야 한다. 다만, 제2항 단서에 따라 지형도면을 작성ㆍ고시하지 아니하는 경우에는 지역ㆍ지구등을 지정할 때에 대통령령으로 정하는 사항을 미리 통보하여야 하고, 제3항 단서에 따라 지역ㆍ지구등의 지정 후에 지형도면등의 고시를 하는 경우에는 지역ㆍ지구등을 지정할 때와 제4항에 따른 지형도면등을 고시할 때에 대통령령으로 정하는 사항을 미리 통보하여야 한다.\n⑨ 제8항에 따라 통보를 받은 시장ㆍ군수 또는 구청장은 그 내용을 국토이용정보체계에 등재하여 지역ㆍ지구등의 지정 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. 다만, 제3항 단서에 따라 지역ㆍ지구등의 지정 후에 지형도면등의 고시를 하는 경우에는 제4항에 따라 지형도면등을 고시한 날부터 일반 국민이 볼 수 있도록 하여야 한다.\n제9조(지역ㆍ지구등의 지정 및 행위제한 내용의 제공)\n① 국토교통부장관과 지방자치단체의 장은 국토이용정보체계를 이용하여 필지별로 지역ㆍ지구등의 지정 여부 및 행위제한 내용을 일반 국민에게 제공하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장은 지역ㆍ지구등이 신설되거나 지역ㆍ지구등에서의 행위제한 내용이 변경되는 경우에는 그 내용을 대통령령으로 정하는 바에 따라 국토교통부장관에게 통보하여야 한다. 이 경우 국토교통부장관은 국토이용정보체계를 통하여 제공되는 내용을 변경하여야 한다. <개정 2013.3.23>\n③ 지방자치단체의 장은 지역ㆍ지구등이 신설되거나 지역ㆍ지구등에서의 행위제한 내용이 변경되는 경우에는 그 내용을 대통령령으로 정하는 바에 따라 국토교통부장관에게 통보하고 국토이용정보체계를 통하여 제공되는 내용을 직접 변경하여야 한다. <개정 2013.3.23>\n제10조(토지이용계획확인서의 발급 등)\n① 시장ㆍ군수 또는 구청장은 다음 각 호의 사항을 확인하는 서류(이하 ""토지이용계획확인서""라 한다)의 발급 신청이 있는 경우에는 대통령령으로 정하는 바에 따라 토지이용계획확인서를 발급하여야 한다.\n② 제1항에 따라 토지이용계획확인서의 발급을 신청하는 자는 시장ㆍ군수 또는 구청장에게 그 지방자치단체의 조례로 정하는 수수료를 내야 한다.\n제11조(규제안내서)\n① 국토교통부장관은 규제안내서를 작성할 수 있다. <개정 2013.3.23>\n② 국토교통부장관이 규제안내서를 작성하려면 관계 행정기관의 장과 미리 협의하여야 한다. 이 경우 협의를 요청받은 관계 행정기관의 장은 특별한 사유가 없으면 그 요청을 받은 날부터 30일 이내에 의견을 제시하여야 한다. <개정 2013.3.23>\n③ 국토교통부장관이 규제안내서를 작성한 경우에는 이를 관보에 고시하여야 하며, 국토이용정보체계를 이용하여 일반 국민에게 제공하여야 한다. <개정 2013.3.23>\n④ 규제안내서에는 다음 각 호의 사항이 포함되어야 한다.\n⑤ 중앙행정기관의 장이 제3항에 따라 고시된 규제안내서에 포함된 내용을 변경하는 경우에는 그 내용을 변경하는 법령의 공포일에 규제안내서의 내용이 변경된 사실과 그 효력 발생일을 함께 관보에 고시하여야 하며, 고시를 하기 전에 미리 고시예정일 등 대통령령으로 정하는 사항을 국토교통부장관에게 통보하여야 한다. 이 경우 국토교통부장관은 국토이용정보체계를 통하여 제공되는 규제안내서를 변경하여 그 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. <개정 2013.3.23>\n⑥ 지방자치단체의 장이 제3항에 따라 고시된 규제안내서에 포함된 내용을 변경하는 경우에는 그 내용을 변경하는 자치법규의 공포일에 규제안내서의 내용이 변경된 사실과 그 효력 발생일을 함께 공보에 고시하여야 하며, 고시를 하기 전에 미리 고시예정일 등 대통령령으로 정하는 사항을 국토교통부장관에게 통보하여야 한다. 이 경우 지방자치단체의 장은 국토이용정보체계를 통하여 제공되는 규제안내서를 변경하여 그 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. <개정 2013.3.23>\n제12조(국토이용정보체계의 구축ㆍ운영 및 활용)\n① 국토교통부장관, 특별시장, 광역시장, 도지사, 시장ㆍ군수 또는 구청장(이하 ""정보체계운영자""라 한다)은 국토의 이용 및 관리 업무를 효율적으로 추진하기 위하여 국토이용정보체계를 구축하여 운영할 수 있다. <개정 2013.3.23>\n② 정보체계운영자는 국토이용정보체계를 통하여 다음 각 호의 사항을 일반 국민에게 제공할 수 있다.\n③ 정보체계운영자는 국토이용정보체계를 효율적으로 만들어 운영하거나 활용하기 위하여 필요하면 전담부서를 설치할 수 있다.\n④ 행정안전부장관 등 관계 행정기관의 장은 제3항에 따라 정보체계운영자가 전담부서를 설치하려는 경우에는 이에 협조하여야 한다. <개정 2013.3.23, 2014.11.19, 2017.7.26>\n⑤ 국토이용정보체계를 통하여 관리되는 정보의 내용과 국토이용정보체계의 구축ㆍ운영 또는 이를 활용한 정보의 제공 및 그 업무 처리에 필요한 사항은 대통령령으로 정한다.\n제13조(지역ㆍ지구등의 지정과 운영 실적 등의 평가)\n① 지역ㆍ지구등을 관장하는 중앙행정기관의 장 및 지방자치단체의 장은 2년마다 지역ㆍ지구등의 지정과 운영 실적 등을 포함한 토지이용규제보고서를 작성하여 국토교통부장관에게 제출하여야 한다. <개정 2013.3.23>\n② 국토교통부장관은 토지이용규제의 적정성을 확보하기 위하여 제22조에 따라 설치된 토지이용규제평가단(이하 ""평가단""이라 한다)으로 하여금 제1항에 따라 제출된 토지이용규제보고서에 기초하여 지역ㆍ지구등의 지정 실태 등을 평가하게 하고, 위원회의 심의를 거쳐 국무회의에 보고한 후 중앙행정기관의 장 또는 지방자치단체의 장에게 그 지역ㆍ지구등의 통합이나 폐합 등 제도개선을 요청할 수 있다. <개정 2013.3.23>\n③ 제2항에 따라 제도개선을 요청받은 중앙행정기관의 장 또는 지방자치단체의 장은 특별한 사유가 없으면 지역ㆍ지구등의 통합이나 폐합 등을 위한 법령 또는 자치법규의 개정 방안, 지역ㆍ지구등을 대체할 수 있는 제도의 신설 등의 대책을 마련하여 국토교통부장관과 협의하여야 한다. <개정 2013.3.23>\n④ 토지이용규제보고서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제14조(행위제한 내용 및 절차에 대한 평가) 국토교통부장관은 서로 다른 지역ㆍ지구등에서 행위제한 내용 및 절차의 균형이 유지되도록 하기 위하여 매년 대통령령으로 정하는 바에 따라 평가단으로 하여금 지역ㆍ지구등에서의 행위제한 내용 및 절차를 조사하여 평가하게 하고, 평가 결과에 대하여 위원회의 심의를 거쳐 중앙행정기관의 장이나 지방자치단체의 장에게 제도개선을 요청할 수 있다. <개정 2013.3.23>\n제15조(토지이용규제심의위원회)\n① 지역ㆍ지구등의 신설 등에 관한 사항을 심의하기 위하여 국토교통부에 토지이용규제심의위원회를 둔다. <개정 2013.3.23>\n② 위원회는 다음 각 호의 사항을 심의한다.\n제16조(위원회의 구성 등)\n① 위원회는 위원장과 부위원장 각 1명을 포함한 20명 이내의 위원으로 구성한다.\n② 위원회의 위원장은 국토교통부장관이 되고, 부위원장은 환경부차관이 된다. <개정 2013.3.23>\n③ 위원장과 부위원장을 제외한 위원은 다음 각 호의 사람이 된다. <개정 2013.3.23>\n④ 위촉위원의 임기는 2년으로 한다.\n제17조(위원의 결격사유)\n① 다음 각 호의 어느 하나에 해당하는 사람은 위원회의 위원이 될 수 없다. <개정 2017.4.18>\n② 위원이 제1항 각 호의 어느 하나에 해당하게 된 때에는 그 날로 위원자격을 잃는다.\n제18조(위원장 등의 직무)\n① 위원회의 위원장은 위원회를 대표하고, 위원회의 업무를 총괄한다.\n② 위원회의 부위원장은 위원장을 보좌하며, 위원장이 부득이한 사유로 직무를 수행할 수 없을 때에는 그 직무를 대행한다.\n③ 위원장과 부위원장이 모두 부득이한 사유로 직무를 수행할 수 없을 때에는 위원장이 미리 지명한 위원이 그 직무를 대행한다.\n제19조(회의의 소집 및 의결정족수)\n① 위원회의 위원장은 위원회의 회의를 소집하고, 그 의장이 된다.\n② 위원회의 회의는 재적위원 과반수의 출석으로 개의(開議)하고, 출석위원 과반수의 찬성으로 의결한다. 다만, 제15조제2항제2호에서 규정한 사항은 재적위원 과반수의 찬성으로 의결한다.\n제20조(간사 및 서기)\n① 위원회에 간사와 서기를 둔다.\n② 간사와 서기는 국토교통부 소속 공무원 중에서 위원장이 임명한다. <개정 2013.3.23>\n③ 간사는 위원장의 명을 받아 위원회의 사무를 담당하고, 서기는 간사를 보좌한다.\n제21조(운영세칙) 위원회의 설치 및 운영에 필요한 사항은 대통령령으로 정한다.\n제22조(토지이용규제평가단)\n① 다음 각 호의 업무를 처리하기 위하여 위원회에 토지이용규제평가단을 설치하여 운영할 수 있다.\n② 평가단의 단장은 위촉위원들이 위촉위원 중에서 단장으로 뽑은 사람이 된다.\n③ 평가단의 구성 및 운영에 필요한 사항은 대통령령으로 정한다.\n제23조(업무의 위탁) 정보체계운영자는 국토이용정보체계의 운영을 대통령령으로 정하는 기관 또는 단체에 위탁할 수 있다.\n제24조(벌칙 적용 시의 공무원 의제) 다음 각 호의 어느 하나에 해당하는 자는 「형법」 제127조 및 제129조부터 제132조까지의 규정을 적용할 때에는 공무원으로 본다.\n\n', 'avg_bleu_4': 0.002763049099007199, 'avg_rouge_1': 0.02844969186838143, 'avg_rouge_l': 0.02844969186838143, 'avg_gpt_score': 1.375}
    {'label': '토지이용규제 기본법', 'full_text': '토지이용규제 기본법\n[시행20170726] [법률 제14839호, 20170726, 타법개정]\n제1조(목적) 이 법은 토지이용과 관련된 지역ㆍ지구등의 지정과 관리에 관한 기본적인 사항을 규정함으로써 토지이용규제의 투명성을 확보하여 국민의 토지이용상의 불편을 줄이고 국민경제의 발전에 이바지함을 목적으로 한다.\n제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다. <개정 2011.4.14>\n\n제3조(다른 법률과의 관계) 지역ㆍ지구등의 지정(따로 지정 절차 없이 법령 또는 자치법규에 따라 지역ㆍ지구등의 범위가 직접 지정되는 경우를 포함한다. 이하 같다)과 운영 등에 관하여 다른 법률에 제8조와 다른 규정이 있는 경우에는 이 법에 따른다.\n제4조(토지이용규제의 투명성 확보) 지역ㆍ지구등을 규정하는 법령 또는 자치법규는 그 지정목적, 지정기준, 행위제한내용 등을 구체적이고 명확하게 규정하여야 한다.\n제5조(지역ㆍ지구등의 신설 제한 등) 지역ㆍ지구등은 다음 각 호에 규정된 것 외에는 신설(지역ㆍ지구등을 세분하거나 변경하는 것을 포함한다. 이하 같다)할 수 없다. <개정 2013.3.23>\n\n제6조(지역ㆍ지구등의 신설에 대한 심의)\n① 중앙행정기관의 장이나 지방자치단체의 장은 지역ㆍ지구등을 신설하는 내용으로 법령 또는 자치법규를 제정하거나 개정하려면 해당 법령안 또는 자치법규안을 입법예고하기 전에 신설될 지역ㆍ지구등이 다음 각 호의 기준에 부합하는지에 대하여 제15조에 따른 토지이용규제심의위원회(이하 ""위원회""라 한다)의 심의를 국토교통부장관에게 요청하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장이나 지방자치단체의 장은 제1항에 따른 심의를 요청할 때에는 지역ㆍ지구등의 지정 및 운영계획서(이하 이 조에서 ""운영계획서""라 한다)를 작성하여 제출하여야 한다.\n③ 국토교통부장관은 제1항에 따른 심의 결과 지역ㆍ지구등의 신설이 제1항 각 호의 기준에 부합하지 아니한다고 인정하는 경우에는 운영계획서를 제출한 중앙행정기관의 장이나 지방자치단체의 장에게 운영계획서의 재검토 또는 수정을 요청할 수 있다. <개정 2013.3.23>\n④ 운영계획서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제6조의2(행위제한 강화등에 대한 심의)\n① 중앙행정기관의 장이나 지방자치단체의 장은 제5조 각 호의 지역ㆍ지구등에서의 행위제한을 신설 또는 강화(이하 ""강화등""이라 한다)하려는 경우에는 해당 법령안 또는 자치법규안을 입법예고하기 전에 다음 각 호의 기준에 부합하는지에 대하여 위원회의 심의를 국토교통부장관에게 요청하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장이나 지방자치단체의 장은 제1항에 따라 심의를 요청할 때에는 행위제한 강화등 계획서(이하 이 조에서 ""계획서""라 한다)를 작성하여 제출하여야 한다.\n③ 국토교통부장관은 제1항에 따른 심의결과 행위제한 강화등이 제1항 각 호의 기준에 부합하지 아니한다고 인정하는 경우에는 계획서를 제출한 중앙행정기관의 장이나 지방자치단체의 장에게 계획서의 재검토 또는 수정을 요청할 수 있다. <개정 2013.3.23>\n④ 계획서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제7조(사업지구에서의 행위제한 등)\n① 개발사업을 시행하기 위한 지역ㆍ지구등(이하 이 조에서 ""사업지구""라 한다)을 규정하는 법령 또는 자치법규는 해당 사업지구에서 개발사업에 지장을 초래할 수 있는 다음 각 호의 행위로서 관계 행정기관의 장의 허가 또는 변경허가를 받아야 하는 사항을 구체적으로 정하여야 한다.\n② 사업지구를 규정하는 법령 또는 자치법규는 다음 각 호의 사항을 구체적으로 정하여야 한다.\n제8조(지역ㆍ지구등의 지정 등)\n① 중앙행정기관의 장이나 지방자치단체의 장이 지역ㆍ지구등을 지정(변경을 포함한다. 이하 같다)하려면 대통령령으로 정하는 바에 따라 미리 주민의 의견을 들어야 한다. 다만, 다음 각 호의 어느 하나에 해당하거나 대통령령으로 정하는 경미한 사항을 변경하는 경우에는 그러하지 아니하다.\n② 중앙행정기관의 장이 지역ㆍ지구등을 지정하는 경우에는 지적(地籍)이 표시된 지형도에 지역ㆍ지구등을 명시한 도면(이하 ""지형도면""이라 한다)을 작성하여 관보에 고시하고, 지방자치단체의 장이 지역ㆍ지구등을 지정하는 경우에는 지형도면을 작성하여 그 지방자치단체의 공보에 고시하여야 한다. 다만, 대통령령으로 정하는 경우에는 지형도면을 작성ㆍ고시하지 아니하거나 지적도 등에 지역ㆍ지구등을 명시한 도면을 작성하여 고시할 수 있다.\n③ 제2항에 따라 지형도면 또는 지적도 등에 지역ㆍ지구등을 명시한 도면(이하 ""지형도면등""이라 한다)을 고시하여야 하는 지역ㆍ지구등의 지정의 효력은 지형도면등의 고시를 함으로써 발생한다. 다만, 지역ㆍ지구등을 지정할 때에 지형도면등의 고시가 곤란한 경우로서 대통령령으로 정하는 경우에는 그러하지 아니하다.\n④ 제3항 단서에 해당되는 경우에는 지역ㆍ지구등의 지정일부터 2년이 되는 날까지 지형도면등을 고시하여야 하며, 지형도면등의 고시가 없는 경우에는 그 2년이 되는 날의 다음 날부터 그 지정의 효력을 잃는다.\n⑤ 제4항에 따라 지역ㆍ지구등의 지정이 효력을 잃은 때에는 그 지역ㆍ지구등의 지정권자는 대통령령으로 정하는 바에 따라 지체 없이 그 사실을 관보 또는 공보에 고시하고, 이를 관계 특별자치도지사ㆍ시장ㆍ군수(광역시의 관할 구역에 있는 군의 군수를 포함한다. 이하 같다) 또는 구청장(구청장은 자치구의 구청장을 말하며, 이하 ""시장ㆍ군수 또는 구청장""이라 한다)에게 통보하여야 한다. 이 경우 시장ㆍ군수 또는 구청장은 그 내용을 제12조에 따른 국토이용정보체계(이하 ""국토이용정보체계""라 한다)에 등재(登載)하여 일반 국민이 볼 수 있도록 하여야 한다.\n⑥ 중앙행정기관의 장이나 지방자치단체의 장은 지역ㆍ지구등의 지정을 입안하거나 신청하는 자가 따로 있는 경우에는 그 자에게 제2항에 따른 고시에 필요한 지형도면등을 작성하여 제출하도록 요청할 수 있다.\n⑦ 제2항에 따른 지형도면등의 작성에 필요한 구체적인 기준 및 방법 등은 대통령령으로 정한다.\n⑧ 중앙행정기관의 장이나 지방자치단체의 장은 제2항에 따라 지형도면등의 고시를 하려면 관계 시장ㆍ군수 또는 구청장에게 관련 서류와 고시예정일 등 대통령령으로 정하는 사항을 미리 통보하여야 한다. 다만, 제2항 단서에 따라 지형도면을 작성ㆍ고시하지 아니하는 경우에는 지역ㆍ지구등을 지정할 때에 대통령령으로 정하는 사항을 미리 통보하여야 하고, 제3항 단서에 따라 지역ㆍ지구등의 지정 후에 지형도면등의 고시를 하는 경우에는 지역ㆍ지구등을 지정할 때와 제4항에 따른 지형도면등을 고시할 때에 대통령령으로 정하는 사항을 미리 통보하여야 한다.\n⑨ 제8항에 따라 통보를 받은 시장ㆍ군수 또는 구청장은 그 내용을 국토이용정보체계에 등재하여 지역ㆍ지구등의 지정 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. 다만, 제3항 단서에 따라 지역ㆍ지구등의 지정 후에 지형도면등의 고시를 하는 경우에는 제4항에 따라 지형도면등을 고시한 날부터 일반 국민이 볼 수 있도록 하여야 한다.\n제9조(지역ㆍ지구등의 지정 및 행위제한 내용의 제공)\n① 국토교통부장관과 지방자치단체의 장은 국토이용정보체계를 이용하여 필지별로 지역ㆍ지구등의 지정 여부 및 행위제한 내용을 일반 국민에게 제공하여야 한다. <개정 2013.3.23>\n② 중앙행정기관의 장은 지역ㆍ지구등이 신설되거나 지역ㆍ지구등에서의 행위제한 내용이 변경되는 경우에는 그 내용을 대통령령으로 정하는 바에 따라 국토교통부장관에게 통보하여야 한다. 이 경우 국토교통부장관은 국토이용정보체계를 통하여 제공되는 내용을 변경하여야 한다. <개정 2013.3.23>\n③ 지방자치단체의 장은 지역ㆍ지구등이 신설되거나 지역ㆍ지구등에서의 행위제한 내용이 변경되는 경우에는 그 내용을 대통령령으로 정하는 바에 따라 국토교통부장관에게 통보하고 국토이용정보체계를 통하여 제공되는 내용을 직접 변경하여야 한다. <개정 2013.3.23>\n제10조(토지이용계획확인서의 발급 등)\n① 시장ㆍ군수 또는 구청장은 다음 각 호의 사항을 확인하는 서류(이하 ""토지이용계획확인서""라 한다)의 발급 신청이 있는 경우에는 대통령령으로 정하는 바에 따라 토지이용계획확인서를 발급하여야 한다.\n② 제1항에 따라 토지이용계획확인서의 발급을 신청하는 자는 시장ㆍ군수 또는 구청장에게 그 지방자치단체의 조례로 정하는 수수료를 내야 한다.\n제11조(규제안내서)\n① 국토교통부장관은 규제안내서를 작성할 수 있다. <개정 2013.3.23>\n② 국토교통부장관이 규제안내서를 작성하려면 관계 행정기관의 장과 미리 협의하여야 한다. 이 경우 협의를 요청받은 관계 행정기관의 장은 특별한 사유가 없으면 그 요청을 받은 날부터 30일 이내에 의견을 제시하여야 한다. <개정 2013.3.23>\n③ 국토교통부장관이 규제안내서를 작성한 경우에는 이를 관보에 고시하여야 하며, 국토이용정보체계를 이용하여 일반 국민에게 제공하여야 한다. <개정 2013.3.23>\n④ 규제안내서에는 다음 각 호의 사항이 포함되어야 한다.\n⑤ 중앙행정기관의 장이 제3항에 따라 고시된 규제안내서에 포함된 내용을 변경하는 경우에는 그 내용을 변경하는 법령의 공포일에 규제안내서의 내용이 변경된 사실과 그 효력 발생일을 함께 관보에 고시하여야 하며, 고시를 하기 전에 미리 고시예정일 등 대통령령으로 정하는 사항을 국토교통부장관에게 통보하여야 한다. 이 경우 국토교통부장관은 국토이용정보체계를 통하여 제공되는 규제안내서를 변경하여 그 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. <개정 2013.3.23>\n⑥ 지방자치단체의 장이 제3항에 따라 고시된 규제안내서에 포함된 내용을 변경하는 경우에는 그 내용을 변경하는 자치법규의 공포일에 규제안내서의 내용이 변경된 사실과 그 효력 발생일을 함께 공보에 고시하여야 하며, 고시를 하기 전에 미리 고시예정일 등 대통령령으로 정하는 사항을 국토교통부장관에게 통보하여야 한다. 이 경우 지방자치단체의 장은 국토이용정보체계를 통하여 제공되는 규제안내서를 변경하여 그 효력이 발생한 날부터 일반 국민이 볼 수 있도록 하여야 한다. <개정 2013.3.23>\n제12조(국토이용정보체계의 구축ㆍ운영 및 활용)\n① 국토교통부장관, 특별시장, 광역시장, 도지사, 시장ㆍ군수 또는 구청장(이하 ""정보체계운영자""라 한다)은 국토의 이용 및 관리 업무를 효율적으로 추진하기 위하여 국토이용정보체계를 구축하여 운영할 수 있다. <개정 2013.3.23>\n② 정보체계운영자는 국토이용정보체계를 통하여 다음 각 호의 사항을 일반 국민에게 제공할 수 있다.\n③ 정보체계운영자는 국토이용정보체계를 효율적으로 만들어 운영하거나 활용하기 위하여 필요하면 전담부서를 설치할 수 있다.\n④ 행정안전부장관 등 관계 행정기관의 장은 제3항에 따라 정보체계운영자가 전담부서를 설치하려는 경우에는 이에 협조하여야 한다. <개정 2013.3.23, 2014.11.19, 2017.7.26>\n⑤ 국토이용정보체계를 통하여 관리되는 정보의 내용과 국토이용정보체계의 구축ㆍ운영 또는 이를 활용한 정보의 제공 및 그 업무 처리에 필요한 사항은 대통령령으로 정한다.\n제13조(지역ㆍ지구등의 지정과 운영 실적 등의 평가)\n① 지역ㆍ지구등을 관장하는 중앙행정기관의 장 및 지방자치단체의 장은 2년마다 지역ㆍ지구등의 지정과 운영 실적 등을 포함한 토지이용규제보고서를 작성하여 국토교통부장관에게 제출하여야 한다. <개정 2013.3.23>\n② 국토교통부장관은 토지이용규제의 적정성을 확보하기 위하여 제22조에 따라 설치된 토지이용규제평가단(이하 ""평가단""이라 한다)으로 하여금 제1항에 따라 제출된 토지이용규제보고서에 기초하여 지역ㆍ지구등의 지정 실태 등을 평가하게 하고, 위원회의 심의를 거쳐 국무회의에 보고한 후 중앙행정기관의 장 또는 지방자치단체의 장에게 그 지역ㆍ지구등의 통합이나 폐합 등 제도개선을 요청할 수 있다. <개정 2013.3.23>\n③ 제2항에 따라 제도개선을 요청받은 중앙행정기관의 장 또는 지방자치단체의 장은 특별한 사유가 없으면 지역ㆍ지구등의 통합이나 폐합 등을 위한 법령 또는 자치법규의 개정 방안, 지역ㆍ지구등을 대체할 수 있는 제도의 신설 등의 대책을 마련하여 국토교통부장관과 협의하여야 한다. <개정 2013.3.23>\n④ 토지이용규제보고서의 작성 및 제출에 필요한 사항은 대통령령으로 정한다.\n제14조(행위제한 내용 및 절차에 대한 평가) 국토교통부장관은 서로 다른 지역ㆍ지구등에서 행위제한 내용 및 절차의 균형이 유지되도록 하기 위하여 매년 대통령령으로 정하는 바에 따라 평가단으로 하여금 지역ㆍ지구등에서의 행위제한 내용 및 절차를 조사하여 평가하게 하고, 평가 결과에 대하여 위원회의 심의를 거쳐 중앙행정기관의 장이나 지방자치단체의 장에게 제도개선을 요청할 수 있다. <개정 2013.3.23>\n제15조(토지이용규제심의위원회)\n① 지역ㆍ지구등의 신설 등에 관한 사항을 심의하기 위하여 국토교통부에 토지이용규제심의위원회를 둔다. <개정 2013.3.23>\n② 위원회는 다음 각 호의 사항을 심의한다.\n제16조(위원회의 구성 등)\n① 위원회는 위원장과 부위원장 각 1명을 포함한 20명 이내의 위원으로 구성한다.\n② 위원회의 위원장은 국토교통부장관이 되고, 부위원장은 환경부차관이 된다. <개정 2013.3.23>\n③ 위원장과 부위원장을 제외한 위원은 다음 각 호의 사람이 된다. <개정 2013.3.23>\n④ 위촉위원의 임기는 2년으로 한다.\n제17조(위원의 결격사유)\n① 다음 각 호의 어느 하나에 해당하는 사람은 위원회의 위원이 될 수 없다. <개정 2017.4.18>\n② 위원이 제1항 각 호의 어느 하나에 해당하게 된 때에는 그 날로 위원자격을 잃는다.\n제18조(위원장 등의 직무)\n① 위원회의 위원장은 위원회를 대표하고, 위원회의 업무를 총괄한다.\n② 위원회의 부위원장은 위원장을 보좌하며, 위원장이 부득이한 사유로 직무를 수행할 수 없을 때에는 그 직무를 대행한다.\n③ 위원장과 부위원장이 모두 부득이한 사유로 직무를 수행할 수 없을 때에는 위원장이 미리 지명한 위원이 그 직무를 대행한다.\n제19조(회의의 소집 및 의결정족수)\n① 위원회의 위원장은 위원회의 회의를 소집하고, 그 의장이 된다.\n② 위원회의 회의는 재적위원 과반수의 출석으로 개의(開議)하고, 출석위원 과반수의 찬성으로 의결한다. 다만, 제15조제2항제2호에서 규정한 사항은 재적위원 과반수의 찬성으로 의결한다.\n제20조(간사 및 서기)\n① 위원회에 간사와 서기를 둔다.\n② 간사와 서기는 국토교통부 소속 공무원 중에서 위원장이 임명한다. <개정 2013.3.23>\n③ 간사는 위원장의 명을 받아 위원회의 사무를 담당하고, 서기는 간사를 보좌한다.\n제21조(운영세칙) 위원회의 설치 및 운영에 필요한 사항은 대통령령으로 정한다.\n제22조(토지이용규제평가단)\n① 다음 각 호의 업무를 처리하기 위하여 위원회에 토지이용규제평가단을 설치하여 운영할 수 있다.\n② 평가단의 단장은 위촉위원들이 위촉위원 중에서 단장으로 뽑은 사람이 된다.\n③ 평가단의 구성 및 운영에 필요한 사항은 대통령령으로 정한다.\n제23조(업무의 위탁) 정보체계운영자는 국토이용정보체계의 운영을 대통령령으로 정하는 기관 또는 단체에 위탁할 수 있다.\n제24조(벌칙 적용 시의 공무원 의제) 다음 각 호의 어느 하나에 해당하는 자는 「형법」 제127조 및 제129조부터 제132조까지의 규정을 적용할 때에는 공무원으로 본다.\n\n', 'avg_bleu_4': 0.002763049099007199, 'avg_rouge_1': 0.02844969186838143, 'avg_rouge_l': 0.02844969186838143, 'avg_gpt_score': 1.375}
    
    


Now we can save the top 10% and bottom 10% for each metric into a new JSON file


```python
percent = float(0.1)
for metric in metrics:
    with open(os.path.join(final_dir, f"avg_{metric}.json"), "r") as avg_file:
        data = json.load(avg_file)
        with open(os.path.join(final_dir, f"top_{str(int(percent*100))}_{metric}.json"), "w") as f:
            json.dump(data[:int(len(data)*percent)], f, indent=2, ensure_ascii=False)
        with open(os.path.join(final_dir, f"bottom_{str(int(percent*100))}_{metric}.json"), "w") as f:
            json.dump(data[-int(len(data)*percent):], f, indent=2, ensure_ascii=False)
```
