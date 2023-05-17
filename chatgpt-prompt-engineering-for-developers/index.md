# ChatGPT Prompt Engineering for Developers


Course Notes of ChatGPT Prompt Engineering for Developers.

<!--more-->


****ChatGPT Prompt Engineering for Developers****

By Andrew Ng+Isa Fulford (OpenAI) 

Original link to the course: [ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)



- Base LLM
- Instruction Tuned LLM

![Untitled](chatgpt-prompt-engineering-for-developers/Untitled.png)

Most applications are based on Instruction Tuned LLM.

**Setup**

```python
import openai
import os
os.environ['OPENAI_API_KEY'] = "sk-" # set your openai api key
openai.api_key = os.environ.get("OPENAI_API_KEY")
# openai.api_base=""

# DEMO
res = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)

res.choices[0]['message']

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

prompt="""XXX"""
response = get_completion(prompt)
print(response)
```

# ****Guidelines for Prompting****

## Prompting Principle **1: Write clear and specific instructions**

### Tactic 1: Use delimiters to clearly indicate distinct parts of the input

- Delimiters can be anything like: `````, `"""`, `< >`, `<tag> </tag>`, `:`

```python
text = f"""xxx"""
prompt = f"""
Summarize the text delimited by triple backticks \ 
into a single sentence.
​```{text}```
"""
```

### Tactic 2: Ask for a structured output

- JSON, HTML

```python
prompt = f"""
Generate a list of three made-up book titles along \ 
with their authors and genres. 
Provide them in JSON format with the following keys: 
book_id, title, author, genre.
"""
```

### Tactic 3: Ask the model to check whether conditions are satisfied

```python
prompt = f"""
You will be provided with text delimited by triple quotes. 
If it contains a sequence of instructions, \ 
re-write those instructions in the following format:

Step 1 - ...
Step 2 - …
…
Step N - …

If the text does not contain a sequence of instructions, \ 
then simply write \"No steps provided.\"

\"\"\"{text_1}\"\"\"
"""
```

### ****Tactic 4: "Few-shot" prompting****

```python
prompt = f"""
Your task is to answer in a consistent style.

<child>: Teach me about patience.

<grandparent>: The river that carves the deepest \ 
valley flows from a modest spring; the \ 
grandest symphony originates from a single note; \ 
the most intricate tapestry begins with a solitary thread.

<child>: Teach me about resilience.
"""
```

## Prompting Principle **2: Give the model time to “think”**

### Tactic 1: Specify the steps required to complete a task

```python
prompt_1 = f"""
Perform the following actions: 
1 - Summarize the following text delimited by triple \
backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the following \
keys: french_summary, num_names.

Separate your answers with line breaks.

Text:
​```{text}```
"""
```

**Ask for output in a specified format**

```python
prompt_2 = f"""
Your task is to perform the following actions: 
1 - Summarize the following text delimited by 
  <> with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the 
  following keys: french_summary, num_names.

Use the following format:
Text: <text to summarize>
Summary: <summary>
Translation: <summary translation>
Names: <list of names in Italian summary>
Output JSON: <json with summary and num_names>

Text: <{text}>
"""
```

### ****Tactic 2: Instruct the model to work out its own solution before rushing to a conclusion****

```python
Question="""XXX"""
Solution="""XXX"""

prompt = f"""
Your task is to determine if the student's solution \
is correct or not.
To solve the problem do the following:
- First, work out your own solution to the problem. 
- Then compare your solution to the student's solution \ 
and evaluate if the student's solution is correct or not. 
Don't decide if the student's solution is correct until 
you have done the problem yourself.

Use the following format:
Question:
```
question here
```
Student's solution:
```
student's solution here
```
Actual solution:
```
steps to work out the solution and your solution here
```
Is the student's solution the same as actual solution \
just calculated:
```
yes or no
```
Student grade:
```
correct or incorrect
```

Question:
```
{Question}
``` 
Student's solution:
```
{Solution}
```
Actual solution:
"""
```

## **Model Limitations: Hallucinations**

Hallucination makes statements that sound plausible but are not true

**Reducing hallucinations:** First find relevant information, then answer the question based on the relevant information.

```python
prompt = f"""
Tell me about AeroGlide UltraSlim Smart Toothbrush by Boie
"""
response = get_completion(prompt)
# Boie is a real company, the product name is not real.
```

# ****Iterative Prompt Development****

![Untitled](chatgpt-prompt-engineering-for-developers/Untitled%201.png)

![Untitled](chatgpt-prompt-engineering-for-developers/Untitled%202.png)

The Prompt workflow involves step-by-step development, similar to model training and tuning. However, success in writing prompts usually requires multiple iterations, as depicted in the figure above. It's important to note that there is no one-size-fits-all perfect prompt. The key to becoming a skilled prompt engineer is having a good development and iterative process for creating prompts.

## Issue 1: The text is too long

Limit the number of words/sentences/characters.

- `Use at most 50 words.`
- `Use at most 3 sentences.`
- `Use at most 280 characters.` (This could be more precise because of the tokenization.)

## **Issue 2. Text focuses on the wrong details**

Ask it to focus on the aspects that are relevant to the intended audience. For example, 

`The description is intended for furnit, so should be technical in nature and focus on the materials the product is constructed from.`

## **Issue 3. Description needs a table of dimensions**

Ask it to extract information and organize it in a table. For example, 

```
After the description, include a table that gives the product's dimensions. The table should have two columns. In the first column include the name of the dimension. In the second column include the measurements in inches only.

Give the table the title 'Product Dimensions'.

Format everything as HTML that can be used in a website. Place the description in a <div> element.
```

# **Summarizing**

```python
prod_review="""xxx"""

prompt = f"""
Your task is to generate a short summary of a product \
review from an ecommerce site to give feedback to the \
pricing deparmtment, responsible for determining the \
price of the product.  

Summarize the review below, delimited by triple \
backticks, in at most 30 words, and focusing on any aspects \
that are relevant to the price and perceived value. 

Review: ```{prod_review}```
"""
response = get_completion(prompt)
```

And try "extract" instead of "summarize”:

If we only want to extract information from a specific aspect, we can request the model to perform "text extraction" instead of "text summarization".

# **Inferring**

Identify types of emotions, Extract useful information, Inferring topics, etc.

```python
prompt = f"""
Identify the following items from the review text: 
- Sentiment (positive or negative)
- Is the reviewer expressing anger? (true or false)
- Item purchased by reviewer
- Company that made the item

The review is delimited with triple backticks. \
Format your response as a JSON object with \
"Sentiment", "Anger", "Item" and "Brand" as the keys.
If the information isn't present, use "unknown" \
as the value.
Make your response as short as possible.
Format the Anger value as a boolean.

Review text: ```{lamp_review}```
"""
response = get_completion(prompt)
```

Infer 5 topics:

```python
prompt = f"""
Determine five topics that are being discussed in the \
following text, which is delimited by triple backticks.

Make each item one or two words long. 

Format your response as a list of items separated by commas.

Text sample: ```{story}```
"""
response = get_completion(prompt)
```

# **Transforming**

Universal Translator, Tone Transformation, Format Conversion, Spellcheck/Grammar check, etc.

Tone Transformation: 

```python
prompt = f"""
Translate the following from slang to a business letter: 
'Dude, This is Joe, check out this spec on this standing lamp.'
"""
response = get_completion(prompt)
```

Format Conversion:

```python
data_json = { "resturant employees" :[ 
    {"name":"Shyam", "email":"shyamjaiswal@gmail.com"},
    {"name":"Bob", "email":"bob32@gmail.com"},
    {"name":"Jai", "email":"jai87@gmail.com"}
]}
prompt = f"""
Translate the following python dictionary from JSON to an HTML \
table with column headers and title: {data_json}
"""
response = get_completion(prompt)
```

# **Expanding**

Customize the automated reply to a customer email

```python
sentiment = "negative"
review = f"""xxx"""
prompt = f"""
You are a customer service AI assistant.
Your task is to send an email reply to a valued customer.
Given the customer email delimited by ```, \
Generate a reply to thank the customer for their review.
If the sentiment is positive or neutral, thank them for \
their review.
If the sentiment is negative, apologize and suggest that \
they can reach out to customer service. 
Make sure to use specific details from the review.
Write in a concise and professional tone.
Sign the email as `AI customer agent`.
Customer review: ```{review}```
Review sentiment: {sentiment}
"""
response = get_completion(prompt)
```

Remind the model to use details from the customer's email

```python
sentiment = "negative"
review = f"""xxx"""
prompt = f"""
You are a customer service AI assistant.
Your task is to send an email reply to a valued customer.
Given the customer email delimited by ```, \
Generate a reply to thank the customer for their review.
If the sentiment is positive or neutral, thank them for \
their review.
If the sentiment is negative, apologize and suggest that \
they can reach out to customer service. 
Make sure to use specific details from the review.
Write in a concise and professional tone.
Sign the email as `AI customer agent`.
Customer review: ```{review}```
Review sentiment: {sentiment}
"""
response = get_completion(prompt, temperature=0.7) 
# 0, choose most likely, 1, randomness+exploration
```

The Temperature parameter affects model diversity by controlling its exploratory and random nature. Lower temperature values (approaching 0) result in stronger determinism while higher temperature values (approaching 1) lead to stronger randomness.

For a stable and reliable system, it is recommended to set the temperature to 0 in a production environment. However, if you seek more creative results, you can set the temperature high.

![Untitled](chatgpt-prompt-engineering-for-developers/Untitled%203.png)

# **Chatbot**

- Dialogues with users (single or multi-character).
- System messages: set the assistant's character and behavior, serve as higher-level commands, and are not perceived by the user.
- Remembering previous conversations: Developers must provide all relevant information for the current conversation. If the model is expected to remember information from previous conversations, the previous dialogue must be input into the model's context.

![Untitled](chatgpt-prompt-engineering-for-developers/Untitled%204.png)

# **Conclusion**

- Two principles:
    - Write clear and specific instructions
    - Give the model time to think.
- Iterative  prompt development
- The capabilities: Summarizing, Inferring, Transforming, Expanding
- Building a chatbot

# Reference

[ChatGPT Prompt Engineering for Developers](https://www.deeplearning.ai/short-courses/chatgpt-prompt-engineering-for-developers/)
