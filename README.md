# HEITS How To - The RAGs Demo

RAGs can be used to give LLMs context about your data (be it text, images or videos). Use this code to start learning more about Retrieval Augmented Generation. 

## How to setup

### 1. Make sure you have Python installed

If you don't have Python installed, you can get it from [this url](https://www.python.org/downloads/). I still use 3.10, but you can try using a higher version.

### 2. Setup the project

Install the dependencies
```
# (Optional) create a virtual environment to store your dependencies

>> python3 -m pip install virtualenv
>> python3 -m venv .venv
>> source .venv/bin/activate

# install requirements

>> pip install -r requirements
```

You will also need to:
* create your own [OpenAI key ](https://help.openai.com/en/articles/4936850-where-do-i-find-my-openai-api-key)
* copy the key 
* create a file called `.env`
* and paste yout key there `OPENAI_API_KEY=YOUR-OPENAI-KEY`

### 3. Run the project

Just type the following command in your console: `python chatbot_ui.py`. This will open a Gradio UI that you can use to play with existing functionalities.