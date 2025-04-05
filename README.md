# Greetings
Hello, if you somehow stumbled upon this project, welcome! I watched this whole video and attempted to learn LangChain and making Generative AI applications. This is the video that I watched:
https://www.youtube.com/watch?v=x0AnCE9SE4A

This video above is from FreeCodeCamp. For context, this is a completion project that is a requirement in one of my subjects. There is a long story behind this, but it's personal matter. Also, the reason for the informative paragraphs is that it is also a requirement for the documentation of this completion project. And, yes, I need to document this project along with references. I also did this individually. Screenshots are used as well instead of the markdown backticks (\`) because it was a requirement for my course. Anyways, I hope that you may find this useful!
# Requirements
To run this application, you'll need to have Conda installed. Conda is a popular package manager that helps you manage dependencies and virtual environments.
## System Requirements
- Python 3.7+
- Conda (for managing dependencies and creating virtual environments)
## Install Conda
If you don't have Conda installed, you can install it by downloading Anaconda or Miniconda:
- [Download Anaconda or Miniconda](https://www.anaconda.com/download/success)
*For this project, I used Anaconda.*
## Setting Up the Environment
1. Clone the repo:
```bash
git clone https://github.com/GeoffJop0908/ai-completion-project
cd <repository-folder>
```
2. Depending on which project you want to try, you can `cd` into them like this example right here:
```bash
cd "1. LangChain Introduction"
```
3. Create a conda environment:
```bash
conda create -p venv python==3.10
```
4. Activate the environment
```bash
conda activate venv/
```
5. Install the `requirements.txt`
```bash
pip install -r requirements.txt -y
```
6. Set the environment variables: This app uses API keys stored in an `.env` file. Make sure to create a `.env` file inside the project folder you are in depending on the required keys of the project.

The general step is following from step 2 and so on to test each project. Whenever you are done on the project folder you are in, you can delete the terminal and open another terminal. An alternative is to run `conda deactivate` to deactivate the environment, and then using `cd` to go to another project directory. When you are in another project directory, make sure to follow from step 2 once again.
# 1. Introduction to LangChain + Q&A Application
## Setup
![Pasted image 20250406014850.png](images/Pasted%20image%2020250406014850.png)

These are the needed keys in your `.env` file. You need your OpenAI API key and your HuggingFace API token. Your HuggingFace tokens are located on your [settings](https://huggingface.co/settings/tokens).

![Pasted image 20250406021029.png](images/Pasted%20image%2020250406021029.png)

This is `requirements.txt`. It lists the dependencies required to run the project. The `langchain` library is a framework for building LLM-powered applications. It simplifies every stage of the LLM development life cycle. It also creates an interface, such as embedding models and vector stores (LangChain, 2025a). We also included the other extensions to the LangChain library such as `langchain_community`, `langchain_openai`, and `langchain_huggingface`. For `openai`, this is OpenAI’s API client for interacting with models like GPT and it provides access to OpenAI's REST API (OpenAI, 2025a). As for `huggingface_hub`, this is a library to access models from Hugging Face. It could access pre-trained models and datasets from HuggingFace (HuggingFace, 2025a). This library will be important in testing a HuggingFace model in our code later on. As for `python-dotenv`, this will be used to load environment variables from our `.env` file (Theskumar, 2025). And then, we have `streamlit`, a framework for building interactive web apps that delivers dynamic data apps (Streamlit, 2025). Lastly, `ipykernel` is another important library for our notebooks. We must install all of these requirements via `pip install -r requirements.txt` as stated earlier.
## Code - `langchain.ipynb`
### Initialization
![Pasted image 20250406020731.png](images/Pasted%20image%2020250406020731.png)
We are going to import OpenAI from the `langchain.llms` module. OpenAI is an American AI research lab comprising a non-profit and a for-profit subsidiary, focused on developing friendly AI, with its systems running on Microsoft's Azure supercomputing platform (LangChain, 2025b). We are then going to import `os` to get access to our `.env` file that contains our environment variables. We are going to use `os.getenv` and the variable name `OPENAI_API_KEY` that contains our API key. Then, we are going to initialize our LLM via `OpenAI()`. We are going to set our temperature to `0.6`

Now, what is temperature? According to IBM et. al (2025), this is the randomness of the generated text. It adjusts the probability distribution of the token selection. Lower temperature means probable tokens will get picked more. This means that the generated content will produce more consistency and precision. This is perfect for tasks like writing documentation. On the other hand, higher temperature means that the likelihood of less probable tokens will get selected. This implies that the generated content will be more random and this is perfect for creative tasks. In our code, we selected just the right temperature, which is `0.6`. 

As for the last block, this will test if the initialized returns no errors. At first, it did return some errors and I had to buy some credits for OpenAI to fix it. It initializes a variable named `text`, which contains our prompt. It contains the question `What is the capital of the Philippines?`. We will then call on the LLM so that it can 'answer' our question using `predict()`. It answered correctly and returned the answer `The capital of the Philippines is Manila.`
### Testing HuggingFace Models
![Pasted image 20250403212603.png](images/Pasted%20image%2020250403212603.png)

This is an old screenshot of the file. I changed the file so that it doesn't have any depreciation warnings. It's practically does the same, but the code in the newer file uses a more direct connection.

This section will test a model from HuggingFace. In this code, we first set our environment variable named `HUGGINGFACEHUB_API_TOKEN` to our HuggingFace access token. We will then import `HuggingFaceHub`. The Hugging Face Hub is a platform that contains models, datasets, and demo apps (which are called spaces) and it is all open source (LangChain, 2025c). We then initialize our HuggingFace LLM with the parameters `repo_id`, `model_kwargs`, and `task`. `repo_id` contains the repository link of our HuggingFace model and our model that we will test is Google's `flan-t5-large`. The FLAN-T5 model is a variant of the T5 (Text-to-Text Transfer Transformer) architecture that has been instruction-finetuned to enhance its performance on various natural language processing tasks (Chung et al., 2022). The `task` parameter contains the task type of the model, which in this case is `text-generation`. Finally, `model_kwargs` contains the additional arguments for our model which is the `temperature` (here, we set it to `0`) and the `max_length` (here, we set it to `64`).

We will then store the output in a variable named `output`. We then use `invoke()` here with our text prompt as a parameter. I used `predict()` first according to the tutorial, but it was a depreciated method and I found a fix on the web for this. We then instructed the model to write a poem about artificial intelligence. But, as you can see, it did not produce such great results and it repeated the phrase "i love you" multiple times. Comparing it to the LLM from OpenAI, it produced a proper poem.
### Prompt Templates
![Pasted image 20250406023216.png](images/Pasted%20image%2020250406023216.png)
According to the documentation, (LangChain, 2025d), prompt templates convert user input and parameters into structured instructions for a language model to ensure relevant and coherent responses. They take a dictionary of variables as input and produce a `PromptValue`, which can be used with an LLM or `ChatModel` and easily switched between string and message formats. In this section, we are going to use prompt templates.

We must import `PromptTemplate` and so that we can structure and format prompts for our language model. We are then going to define a prompt template using our variable `prompt_template`. We are then going to use `PromptTemplate` that has a parameter of `input_variables` which contains our placeholder variables and `template` which is our prompt itself. `input_variables` contains an array of only one item, which is `country`. The placeholder will be replaced with actual values when formatting. We will then fill in `{country}` with `"Philippines"` using `format()` which will result in a string like this: `"Tell me the capital of Philippines"`. 

Originally, the video used the `LLMChain`. But, in the newer versions, the `|` pipe operator is used for better readability. Chains help connect our LLM with our prompt template. We will use chains to create our `chain` with `llm` that contains the LLM we created earlier, and `prompt_template`. Here, we used the pipe operator to create our `chain`. We can run this chain with a parameter of `"Philippines"` by using `invoke()`. Since we have a template, the LLM will process the prompt `"Tell me the capital of Philippines"` and will output `"The capital of Philippines is Manila"`.
### Simple Sequential Chain

![Pasted image 20250406024549.png](images/Pasted%20image%2020250406024549.png)
At first, we just copied what we did earlier in our prompt templates section which asks the LLM in a format that would ask what the capital of a country is using `PromptTemplate` and chains. We will just use a different variable called `capital_prompt`. We will then use another prompt template, but this time it asks what are the famous places in that capital. We stored the chains in each variable named `capital_chain` and `famous_chain`. We then combine these chains inside `combined_chain`. 

Originally, we used `SimpleSequentialChain`. This chain was a straightforward sequence where the output of one step directly serves as the input for the next (LangChain, 2025e). But again, we used the pipe operator as that is the new way of creating chains. The output of `capital_chain` becomes the input for the second chain, which is `famous_chain`. We can run the chain using `combined_chain.invoke()` with `"Philippines"` as our parameter once again. The output is a list of famous places to visit in Manila, which is the capital of the Philippines. The list includes popular tourist destinations such as Intramuros, Rizal Park, and Manila Ocean Park.
### Sequential Chain
Simple sequential chains are the simplest form of sequential chains. Each step in this chain has a singular input or output and the output of one step is the input to the next chain. Meanwhile, sequential chains is the general form. It has multiple inputs or outputs (LangChain, 2022). 

In this section, the code demonstrated how to create a multi-step processing pipeline that originally used the deprecated `SequentialChain`. This code represents the modern equivalent of sequential chains. It takes a country as input, retrieves its capital using the LLM, and then uses that capital to generate a list of famous places to visit. Since simple sequential chain did not provide us the output on each chain, this code will provide both the outputs of each chain.

Here, I will include both the old screenshot that uses the deprecated modules and the new screenshot that uses the new syntax. I will also include the explanation of each code.

#### Old Deprecated `SequentialChain` and `LLMChain`
![Pasted image 20250403221012.png](images/Pasted%20image%2020250403221012.png)
We used `PromptTemplate` here once again so that our prompt uses a template. We copied it from our previous code that demonstrated simple sequential chains. We stored our prompt templates into `capital_prompt` and `famous_template` and our chains into `capital_chain` and `famous_chain`. The difference now is that we used the parameter `output_key` which will set its key in the dictionary output. For example `capital_chain` will produce an output of `{"capital": "Manila"}`. 

We will then import `SequentialChain` and use that as our `chain`. Our parameters contains `chains`, `input_variables`, and `output_variables`. `chains` defines the sequence in which the chains are executed. `input_variables` contains the input variable for the chain, which is `'country'`. Lastly, `output_variables` is the final output which contains both the capital city and the recommended places.

For running the chain, instead of using `run()` like before, we will now directly call it as if it is a function. The input `"Philippines"` is passed to the chain. The first chain retrieves the capital: `"Manila"` and this will be passed to the second chain and get a list of famous places in Manila. The output then contains a dictionary that contains the keys `country`, `capital`, and `places`. Each contains the country input that we put earlier, the capital, and the recommended places in the capital respectively.
#### New Modern Equivalent
![Pasted image 20250406025836.png](images/Pasted%20image%2020250406025836.png)
![Pasted image 20250406025847.png](images/Pasted%20image%2020250406025847.png)

Full output:
```
{
	'country': 'Philippines',
	'capital': '\n\nThe capital of Philippines is Manila.',
	'places': " Here are some amazing places to visit in Manila:\n1. Intramuros - the historic walled city that showcases the Spanish colonial architecture and culture of Manila.\n2. Rizal Park - a large urban park that features gardens, monuments, and the Rizal Monument, a tribute to the national hero.\n3. National Museum of the Philippines - the country's premier museum that houses a vast collection of art, artifacts, and natural history exhibits.\n4. Manila Ocean Park - an oceanarium and theme park that offers a unique underwater experience with various marine animals.\n5. Binondo - the oldest Chinatown in the world, known for its bustling markets, authentic Chinese food, and cultural landmarks.\n6. Bonifacio Global City - a modern and upscale district with shopping centers, restaurants, and entertainment venues.\n7. Fort Santiago - a historic citadel located within Intramuros, with a museum and park that showcases Manila's past.\n8. Manila Bay - a natural harbor that offers stunning views of the sunset and is a popular spot for dining and leisure activities.\n9. San Agustin Church - a UNESCO World Heritage Site and the oldest stone church in the Philippines, known for its beautiful Baroque architecture.\n10. Ayala Museum - a museum that showcases the"
}
```

Here, we still kept the `PromptTemplate` from LangChain. Thus, our `capital_prompt` and `famous_template` remained the same. We defined two functions that would replace `LLMChain`. The `get_capital` function accepts an inputs dictionary containing a `'country'` key.​ It extracts the value associated with `'country'`.​ The `capital_prompt` is formatted with the extracted country to create a specific prompt.​ The prompt is then passed to the `llm` and uses `invoke()` to generate a response.​ The function returns the response in a dictionary with the key 'capital'.

This is the same for `get_places()`. We extract the value that was associated with `capital` from the input and then we pass this to `famous_template` to format it as the prompt template. We then pass this to the `llm` and put the `response` as a value to the key named `places`.

The `chain.invoke()` method is then called with an input dictionary specifying the country as `'Philippines'`.​ This initiates the sequence of prompts defined in the chain, processing the input through each function in order.​ The chain is assumed to be a sequence of runnables (functions or models) that process the input data step by step.

To recreate the final output dictionary from the old one, we will define a `final_result` dictionary. It includes the original country (`'Philippines'`) although this is hardcoded and may be improved by using a variable. The other contents of the dictionary include the capital retrieved from the `result`, and the places to visit in that capital.​

We will then `print()` the `final_result`. It displays the structured information about the country, its capital, and suggested places to visit.
#### Difference
Although deprecated, I will still explain the difference of sequential chains and simple sequential chains. This difference is that sequential chains allows for more flexibility. The simple sequential chain takes a single input and passes it sequentially while the sequential chain allows multiple input variables. We also demonstrated the flexibility of this chain here by printing both the capital and places to visit, which we did not get with simple sequential chains. Sequential chains also output dictionaries for structured output while simple sequential chains passes the results directly. Each chains have different use cases. If you only need the final result and a quick, linear execution without extra complexity, use simple sequential chains. If you need intermediate results (such as the capital and the places) and require structured output with named keys, use sequential chains.
### Chat Models with ChatOpenAI
![Pasted image 20250406031016.png](images/Pasted%20image%2020250406031016.png)
According to the documentation, ChatOpenAI is a wrapper for OpenAI's chat-based language models which allows structured interaction using a message-based format. It supports different message types, including SystemMessage, HumanMessage, and AIMessage, which define roles in the conversation (LangChain, 2025f). In this section, we will set up a chatbot that responds in a Tagalog conyo comedian style.

We start by importing essential modules: `ChatOpenAI`, `HumanMessage`, `SystemMessage`, `AIMessage` and `os`. `ChatOpenAI` initializes an OpenAI-powered chatbot and uses ChatGPT while `HumanMessage`, `SystemMessage`, and `AIMessage` define the different types of messages in the conversation. `os` is used to retrieve the API key stored in environment variables.

We then instantiate `ChatOpenAI` inside a variable named `chat_llm`. The parameters that we used are `openai_api_key`, `temperature`, and the `model`. We retrieve the OpenAI API key from our `.env` file using `os.environ` and pass it to our `openai_api_key` parameter. We then set our `temperature` to `0.6`, which is just the right amount. We then used the specific model of ChatGPT, which is GPT4o that is a version from 2024-08-06.

Once we instantiated that in our `chat_llm` variable, we will define a conversation using structured messages. We will pass in an array to `chat_llm` that contains the items, `SystemMessage` and `HumanMessage`. `SystemMessage` sets the chatbot’s role/personality. In this case, it’s a Tagalog conyo comedian AI. The `HumanMessage` represents the user’s message requesting punchlines regarding AI. Make sure to use `invoke()` to call the AI. This will then return an `AIMessage` class that returns the content of the message and its metadata. This is the whole content of the message:

```
Sure, bro! Eto ang ilang AI punchlines:

1. Bakit hindi marunong mag-surfing ang AI? Kasi lagi siyang off the cloud!

2. Anong sabi ng AI sa kanyang crush na chatbot? "Kahit anong algorithm, ikaw pa rin ang aking end-goal!"

3. Bakit hindi puwedeng maging stand-up comedian ang AI? Kasi lahat ng jokes niya, scripted!

4. Paano mo malalaman kung AI ang kausap mo? Kapag sinabi niyang, "I'm feeling lucky," pero wala naman talaga siyang feelings!

5. Anong sabi ng AI sa kanyang best friend na human? "Bro, ikaw na lang ang may puso, kaya ikaw na rin ang magbayad!"

Sana natawa ka kahit konti, dude!
```
### Prompt Template + LLM + Output Parsers
![Pasted image 20250406031512.png](images/Pasted%20image%2020250406031512.png)
`ChatPromptTemplate` is a prompt template designed for chat models, and it allows the creation of flexible, templated prompts for conversational AI models (LangChain, 2025g). Meanwhile, `BaseOutputParser` serves as the base class for parsing the output of a language model (LLM) call and helps to structure the responses generated by the model (LangChain, 2025h). While `PromptTemplate` is best suited for generating single-string prompts for general language modeling tasks, `ChatPromptTemplate` is optimized for creating structured prompts that represent conversational exchanges in chat-based applications. In this section, we create a pipeline that takes a word as input and returns five synonyms in a comma-separated format.

We will import the necessary components: `ChatOpenAI`, `ChatPromptTemplate`, and `BaseOutputParser`. Respectively, these components initializes the AI-powered chatbot, helps format structured prompts for conversational models, and defines how the model’s output is processed. These libraries are important for this section.

Next, we are going to define a custom output parser using our `CommaSeparatedOutput` class. This class will inherit from the `BaseOutputParser`. We will overwrite `parse()` so that we can remove extra spaces using `strip()` and split the string into a list via commas using `split(",")`. This ensures that the output is structured as an array of synonyms rather than a single string..

We now create a prompt template that structures how the AI should respond using `template`, `human_template` and `ChatPromptTemplate`. `template` sets up the AI’s behavior so that it must return five synonyms in a comma-separated format. This will be our constraint to the AI. The human message, `human_template`, is a placeholder `{text}` that will be replaced by the user’s input. We then structure these messages into a chat-based format using `ChatPromptTemplate`. `from_messages()` arranges the system and human messages into an interactive format.

​In LangChain, the pipe operator (|) is utilized to create a sequence of operations, known as a `RunnableSequence`, where the output of one component serves as the input to the next. This chaining mechanism enhances code readability and efficiency by streamlining data flow between functions or models (LangChain, 2025i). Here, we used it to chain the prompt, model, and output parser together. `chat_prompt` formats the input, `chat_llm` (the AI model) generates a response, and the `CommaSeparatedOutput()` parses the response into a structured list.

Finally, we invoke the pipeline with an example input using `invoke()`. We pass in a dictionary that contains a `text` key with our value `fragrant` as an example input. The LLM will then output the array of synonyms: `["aromatic", "perfumed", "scented", "redolent", "fragrant"]`.
## Code - `app.py`
![Pasted image 20250406031900.png](images/Pasted%20image%2020250406031900.png)
This code demonstrates how to build a simple Q&A chatbot application using LangChain for language modeling and Streamlit for creating an interactive web interface. The chatbot uses OpenAI’s GPT model to respond to user queries. This creates an easy user interface for our app. We can run this app via `streamlit run app.py` and it will automatically open your browser and run the app.

We first import our necessary libraries: `ChatOpenAI`, `dotenv`, `os`, and `streamlit`. Once again, `ChatOpenAI` is used so that we can interact with the OpenAI chat models, while `dotenv` and `os` is used for managing our environment variables that were loaded on our `.env` file. `streamlit` is another necessary library for this and this is a framework used to create an interactive user interface for the chatbot.

We then load our environment variables via `load_dotenv()`. This function loads the variables from our `.env` file, which makes them accessible within the Python environment. The `.env` file is expected to contain the OpenAI API key and it must be stored in our `OPENAI_API_KEY` environment variable.

We then define the function to get responses from OpenAI. `get_openai_response()` is a function that accepts a `question` argument. It initializes a `ChatOpenAI` model with the provided API key that was retrieved from the environment variable and a specific model version of GPT-4o. It will then send a question to the model and it is structured as a message with the `"role": "user"` and the `content` being the `question` parameter. It will then return the model's response, which is accessed from the `response.content`.

We will then initialize the Streamlit app for the web interface. `st.set_page_config()` configures the Streamlit web page with a custom title ("QnA Demo"). The `st.header()` adds a header to the webpage. We then have an input box using `st.text_input()`. It displays a text input box where users can type their question. The input value is stored in `input_text`. The `key="input"` ensures Streamlit keeps track of the input’s state across interactions.

We then handle input validation in our `if` block and generate our response there. `if input_text` checks if the user has typed something into the input box. If the user has entered a question, the function `get_openai_response()` is called and it passes the user’s question `input_text` to the OpenAI model. The chatbot’s response is displayed in Streamlit using `st.subheader()`, which adds a subheader before displaying the response and `st.write()` which outputs the model’s response.
## App
![Pasted image 20250404162245.png](images/Pasted%20image%2020250404162245.png)
Here the app is running on `localhost` on port `8501`. This app is ran by using the command `streamlit run app.py`. Make sure to install the `requirements` first before running.

![Pasted image 20250404162534.png](images/Pasted%20image%2020250404162534.png)
Here, the app was tested by asking the AI a question. The question was "Who was Jose Rizal?" The AI responded with detailed information about him.
## Deploying the App on HuggingFace
![Pasted image 20250404162643.png](images/Pasted%20image%2020250404162643.png)
Here, the app is deployed in HuggingFace. To deploy this app, head on over to `https://huggingface.co/spaces`. Then, at the top of the search bar, you will see a button that says "New Space" with a plus button.

![Pasted image 20250404162825.png](images/Pasted%20image%2020250404162825.png)
![Pasted image 20250404162836.png](images/Pasted%20image%2020250404162836.png)
Once you arrived at this screen, you can set your configurations. You can change the Space name into anything you want. For my app, I used the name `langchain-qna-demo` . Then, you can skip the description and the license. Choose the Streamlit button because that will be our SDK. We can use the free CPU for the space hardware. We can choose to make the space private. You can then now hit the "Create Space" button once you are done configuring.
![Pasted image 20250404163235.png](images/Pasted%20image%2020250404163235.png)
Once the app is done initializing, you must head on over to the Files section. Here, you must upload the files by clicking the Add file button on the top right. On the dropdown, make sure to click Upload Files. You can drag and drop the `app.py` file and the `requirements.txt` file and the space will automatically initialize this app for you.
![Pasted image 20250404163514.png](images/Pasted%20image%2020250404163514.png)
![Pasted image 20250404163529.png](images/Pasted%20image%2020250404163529.png)
Testing the app, it works. Here we asked the AI to give information about Mapua University. The AI responded with a numbered list and Streamlit is handling the markdown format.
# 2. Chat With PDF Using LangChain And AstraDB
LangChain also provides various modules for integrating vector storage, indexing, language modeling, and text embedding. The `Cassandra` module allows efficient storage and retrieval of vector embeddings using Apache Cassandra. It is also a NoSQL, row-oriented, scalable, and available database that has vector search capabilities. It requires the `cassio` library and a compatible database. (LangChain, 2025j). `VectorStoreIndexWrapper` simplifies interaction with vector stores to enable easier management of document embeddings and similarity searches (LangChain, 2025k). The `OpenAI` module facilitates integration with OpenAI's language models for tasks like text generation, requiring the `openai` package and API key (LangChain, 2025b). Similarly, `OpenAIEmbeddings` enables the generation of vector representations of text for semantic search and clustering (LangChain, 2025l). Together, these modules enhance LangChain’s capabilities for NLP and AI-driven applications. We will use this capability to chat with a PDF with LangChain for this section.
## Database Setup on DataStax
![Pasted image 20250404174721.png](images/Pasted%20image%2020250404174721.png)
Here, we head on to DataStax. ​DataStax is a real-time AI company that empowers developers and enterprises to build generative AI applications at scale. Their flagship product, Astra DB, is a vector database built on Apache Cassandra which is designed to support production-level AI applications with real-time data (DataStax, 2025). We will use this as a database to store our vectors for stateless management.

First, you must create your account first and you can either use your Google or GitHub account with this. For me, I used my Google account since I encountered some problems on my GitHub account. We can then now create our own database. In the configuration modal, you must select the Serverless (Vector) deployment type. You can then set the database name to anything you want. Here, I entered `pdfquery` as my database name. You must set your provider to Google Cloud and the region to the only option available, which is US east.

![Pasted image 20250404175417.png](images/Pasted%20image%2020250404175417.png)

In this section, you can see the database details. This is located on the left side of the screen. You must generate an application token by clicking on the Generate Token button. You must copy this token and save it somewhere as it will be important later on.
## Code
### Installation of Libraries
![Pasted image 20250404165052.png](images/Pasted%20image%2020250404165052.png)
Here, we are going to install dependencies. We must first upgrade `pip`, `setuptools`, and `wheel`. I've done this in Google Colab and I did this command to ensure the compatibility between the modules. We then install the necessary libraries like `cassio`, `datasets`, `langchain`, `openai`, and `tiktoken`. We all know what the `langchain` and `openai` library does, but what are the other unfamiliar ones? First, `cassio` is for our Cassandra database. Next, the `datasets` library is a lightweight, efficient, and scalable tool for downloading, preprocessing, and handling datasets across various machine learning modalities that came from HuggingFace (2025b). We used `load_dataset` in the code, but, it was never used since we already have a sample file named `research_paper.pdf`. Although, this library would be handy in loading some datasets from HuggingFace for a sample PDF file. Lastly, `tiktoken` is a fast Byte Pair Encoding (BPE) tokenizer developed by OpenAI for use with their language models (OpenAI, 2025b). We also installed `langchain_community` so that it would not return an error later in our code. These are all important libraries to make our app work. 
![Pasted image 20250406032809.png](images/Pasted%20image%2020250406032809.png)
Here, we imported all the important stuff from LangChain. We also imported `PdfReader` to allow us to read PDF files in python. It came from a library called `PyPDF2`. This library is free and open-source and enables users to manipulate PDF files by performing tasks such as splitting, merging, cropping, and transforming pages (Martin.Thoma & mstamy2, 2022).
### Token setup
![Pasted image 20250406032923.png](images/Pasted%20image%2020250406032923.png)
Here, we set up our tokens and IDs. We will put our AstraDB Tokens so that we can authenticate with the Cassandra vector database. We will store them in a variable named `ASTRA_DB_APPLICATION_TOKEN` (the access token to our database) and `ASTRA_DB_ID` (the ID of the database). We then store our OpenAI API Key to access our LLM. We used `os.getenv()` to retrieve the environment variables.

We then use `PdfReader` from the `PyPDF2` library and pass in on our `research_paper.pdf`. The paper that was used here was a paper by Jin et al. (2024) that discusses about the usage of graph neural networks for time series. We then read each page using `enumerate()` and extract text using `extract_text()`. It will then check if `content` is empty and store the full text in `raw_text` if `content` is not empty.
![Pasted image 20250406033038.png](images/Pasted%20image%2020250406033038.png)

Here, we printed `raw_text`. We can see the contents of it. It contains the text that was retrieved from the PDF.
### Database Connection and LLM Instantiation
![Pasted image 20250406033133.png](images/Pasted%20image%2020250406033133.png)
We then initialize our AstraDB connection using `cassio.init()` and passing our application tokens and database ID. We then create an OpenAI LLM instance via the `llm` variable and pass in our OpenAI API key. We must also create an embedding model via the `embedding` variable for text processing and pass in our key as well.
### Vectorization
![Pasted image 20250406033155.png](images/Pasted%20image%2020250406033155.png)
We then create a Cassandra-based vector store using `Cassandra` class. We will pass our `embedding`, a `table_name` which is named `qa_demo`, and `None` to `session` and `keyspace`. We store our vector embeddings in this database for efficient semantic search.

We will then use the `CharacterTextSplitter` from LangChain to split our text. We will split the extracted text into chunks of 800 characters. This is the value that we passed on to `chunk_size` parameter. We pass `200` to `chunk_overlap` because overlapping 200 characters ensures better context retention. We then store all of the split text to our `texts` variable and we print out the first 50 chunks using `texts[:50]`.

![Pasted image 20250406033210.png](images/Pasted%20image%2020250406033210.png)

We then embed and store all the text chunks using `add_text` method on our Astra vector store. We must confirm this insertion with a print statement. We will then create an indexed wrapper around `astra_vector_store` for efficient querying.
### Running the Q&A Loop
![Pasted image 20250404173952.png](images/Pasted%20image%2020250404173952.png)
![Pasted image 20250404174004.png](images/Pasted%20image%2020250404174004.png)
Here, we used a `while` loop to run the Q&A cycle. The user must input a question. If they typed "quit", the loop exits. If they typed an actual question, it queries the vector index using `astra_vector_index.query(query_text, llm=llm)`. It then performs similarity search with `k=4` to find the most relevant text in the document. We can then print the AI’s response to the user's question.

The expected output can be seen above. One of the questions that was asked was "What are the advantages of GNNs?" and the AI returned the most relevant texts in the document. It displayed the number on how "relevant" the text was to the question. It also displayed the text that was relevant.
# 3. Blog Generation Using LLAMA LLM Models
As for the third project, we will be creating a blog generation app using Streamlit once again. In the video, LLAMA 2 was used. But for my code, I will be using the GPT-4o model from OpenAI, so there will be minor tweaks to this app. The LLAMA model is also slow in response, so this is where the GPT model can shine through more. This app takes a blog topic, word limit, and target audience as inputs and returns an AI-generated blog.
## Setup
![Pasted image 20250404224940.png](images/Pasted%20image%2020250404224940.png)
This is the contents of `requirements.txt`. It contains the various required libraries like `langchain`, `openai`, and `streamlit`. We need all of this for our blog generation app.

![Pasted image 20250404225032.png](images/Pasted%20image%2020250404225032.png)
This is the `.env` file. It contains our OpenAI API key. This will be necessary to call the LLM.
## Code
### Imports
![Pasted image 20250406033322.png](images/Pasted%20image%2020250406033322.png)
This section imports the libraries necessary for our app. `streamlit` (`st`) is used to create an interactive web app while `PromptTemplate` from `langchain.prompts` helps format dynamic prompts for GPT-4o. `ChatOpenAI` from `langchain_openai` provides access to GPT-4o for generating responses. Finally, `os` allows us to manage the environment variables where our API keys are located.

![Pasted image 20250404225648.png](images/Pasted%20image%2020250404225648.png)
This function takes three inputs: `input_text`, `no_words`, and `blog_style`. `input_text` will be the blog topic, `no_words` will be the word limit, and `blog_style` will be the intended audience. We will then initialize GPT-4o using `ChatOpenAI()` with the `model` being `gpt-4o`.

The prompt template structures the input dynamically by using placeholders. `{blog_style}` will be the writing style based on the target audience. `{input_text}` will be the blog topic. Finally, `{no_words}` will be the word limit.

We will then fill in the placeholders with user inputs using `prompt.format()`. Then, we will send the formatted prompt to GPT-4o to generate a response using `llm.invoke()`. Finally, we must return the message via `response.content` for the final blog output.

![Pasted image 20250404230158.png](images/Pasted%20image%2020250404230158.png)

We will set up the Streamlit page, using `st.set_page_config()`. The title would be "Generate Blogs" with the icon being this emoji: 📝. The icon will set the favicon of the tab. This app will be in a centered layout with a collapsed sidebar. We will also display the header at the top of the app.

We will set up a text input field via `st.text_input` to allow the user to enter the blog topic. We are then going to create two equal columns, `col1` and `col2`, for the additional inputs. Inside the 1st column, `col1`, it would contain our variable `no_words`, which is a text input field for word count. For the 2nd column, we will contain a variable named `blog_style`, which is a dropdown (`selectbox`) to allow the user to choose the target audience.

We will then create a button (`submit`) to trigger the blog generation process when clicked. When this submit button is clicked the `get_gpt4o_response()` function is called with the user inputs. The generated blog is then displayed on the page using `st.write()`.
## App
![Pasted image 20250404230859.png](images/Pasted%20image%2020250404230859.png)
![Pasted image 20250404230914.png](images/Pasted%20image%2020250404230914.png)

Here, we tested a sample topic. The audience is for the general people and the blog must have 300 words. The topic that I entered here was "What do people think about Filipino Food?" and The AI generated a blog post.

![Pasted image 20250404231058.png](images/Pasted%20image%2020250404231058.png)
![Pasted image 20250404231126.png](images/Pasted%20image%2020250404231126.png)

Here, we generated the same topic but for a different audience. The audience now is for researchers. As you can see, the words are more formal and is catered to researchers.

![Pasted image 20250404231303.png](images/Pasted%20image%2020250404231303.png)

Here, we can see the different options. The video originally had three options, specifically the researchers, data scientist, and general people. I added two more options, which are the other bloggers and market professionals. This is to ensure that I can test the AI's flexibility.
# 4. End To End LLM Projects Using Pinecone VectorDB
For some reason, we are going to chat with a PDF again. This time we are going to use Pinecone instead of AstraDB and DataStax. I was promised a quiz app, but all I get is another PDF chat example. 

Anyways, let's discuss about Pinecone now. Pinecone is a fully managed, cloud-native vector database designed to handle high-dimensional vector data efficiently. This enables us to do rapid similarity searches and real-time machine learning applications. It abstracts the complexities of managing underlying infrastructure which makes it ideal for applications like semantic search, recommendation systems, and anomaly detection (MLJourney, 2024).

Now, ​DataStax and Pinecone are both vector databases used for storing and retrieving vector embeddings, but they differ in performance, features, and suitability for various applications. DataStax, built on Apache Cassandra, provides a comprehensive data platform that supports real-time analytics, high availability, and scalability. This is ideal for large-scale applications in industries like finance and retail. It outperforms Pinecone in terms of data ingestion, indexing throughput, and query response times. Therefore DataStax is more suitable for more complex applications. In contrast, Pinecone is a cloud-native, managed vector database optimized for similarity searches and machine learning applications, particularly useful for AI-driven search engines and recommendation systems (Norrid, 2025). 

Each platform has their own use cases. DataStax is suited for a wide range of data management tasks, while Pinecone excels in high-dimensional vector data storage and retrieval for specialized machine learning applications. Now that is out of the way, let's review the code!
## Setup
![Pasted image 20250405043322.png](images/Pasted%20image%2020250405043322.png)
First, let's setup our Pinecone database. You must create your account in Pinecone first. Here, I used my Google account once again. I already have my index, but, we can ignore that for now. You should still proceed and create a new index by pressing on the Create Index button.

![Pasted image 20250405043513.png](images/Pasted%20image%2020250405043513.png)
![Pasted image 20250405043616.png](images/Pasted%20image%2020250405043616.png)
This screen should appear. You can set your index name to anything you want. As for the configuration, you should tick the checkbox named "Custom settings" so that we can have our own custom dimensions. Here, I entered 1536 as my dimension, which I will explain later on. I did not change anything with the vector type and the metric, so they should be the same.

![Pasted image 20250405043749.png](images/Pasted%20image%2020250405043749.png)
![Pasted image 20250405043803.png](images/Pasted%20image%2020250405043803.png)
![Pasted image 20250405043854.png](images/Pasted%20image%2020250405043854.png)
The capacity mode would be serverless since that would be our only option. As for the cloud provider, I chose amazon's AWS. As for the region, I don't have no choice but to pick the only available server, which is Virginia. After that, don't touch anything with the deletion protection (unless you want to) and proceed by creating the index by pressing the Create Index button. Now, you have created your vector database. Let's proceed to the code. 
## Code
### `requirements.txt`
![Pasted image 20250405044759.png](images/Pasted%20image%2020250405044759.png)
Here are the necessary requirements so that our notebook will work. We all know what the LangChain and OpenAI library does, so, let's explain what the other libraries are. The `unstructured` is an open-source library that provides components for ingesting and pre-processing various document types, especially PDFs. This will be a useful library for facilitating streamlined data processing workflows for LLMs (Unstructured-Io, 2025). `tiktoken` is a familiar library, but to recap, it is a tokenizer that was developed by OpenAI to encode our text. The `pinecone` library will be the SDK and this will be necessary for doing actions to the Pinecone database. `pypdf` and `PyPDF2` is another familiar library that we used earlier and this will be used in doing different actions to our PDF file. `pandas` and `numpy` are essential libraries for machine learning and developing AI applications. `pandas` is great for working datasets (W3Schools, 2025) while `numpy` is the fundamental library for scientific computing in Python (NumPy Developers, 2024). `python-dotenv` is another library that will be useful once again to access our environment variables. Lastly, `ipykernel` is a package that provides the IPython kernel for Jupyter notebooks (IPython development team, 2025).
### `.env`
![Pasted image 20250405050419.png](images/Pasted%20image%2020250405050419.png)

As for our `.env` file, it must contain two important keys. First, you must enter the `OPENAI_API_KEY`, which is our OpenAI API key. Second is the Pinecone API key that was created when we set up the database.
### Importing Relevant Modules
![Pasted image 20250406033837.png](images/Pasted%20image%2020250406033837.png)
Now, let's head on to the code. First, we must import all the important libraries and modules. We must import `Pinecone` from the `pinecone` library to connect the client to the database. We then import `PyPDFDirectoryLoader`, which will be the module responsible for loading and parsing of all PDF files within a specified directory. It utilizes the `pypdf` library to extract text from PDFs and metadata (LangChain, 2025m). `RecursiveCharacterTextSplitter` is a text splitter that recursively divides text into smaller chunks based on a list of specified separators. This is helpful for processing long documents (LangChain, 2023a). `OpenAIEmbeddings` is what we've used before to transform text into vectors. Lastly, `PineconeVectorStore` is part of the `langchain_pinecone` package and facilitates the integration of LangChain with Pinecone (LangChain, 2023b).

We will also import `load_dotenv` from `dotenv`. This is important as it will load all the environment variables from `.env`. We the also import `os` so that we can retrieve these loaded variables.
### Reading the PDF
![Pasted image 20250405051714.png](images/Pasted%20image%2020250405051714.png)
![Pasted image 20250405051726.png](images/Pasted%20image%2020250405051726.png)

Here, we defined a `read_doc` function to read the documents in our `documents` directory. We used the `PyPDFDirectoryLoader` to load the documents and store it in `file_loader`. We then `load()` this and store it in `documents`.  This returns an array of `Document` objects. We called this function on `doc` and now this variable will store all of the `Document` objects. We printed out the output which is an array. We also printed out the length of this array which is the number of pages of the document. Here we have a 34-page document. The document that was used here was a research paper by Chopra and Sharma (2021) that discusses about the different applications of artificial intelligence in the stock market.
### Splitting the Documents into Chunks
![Pasted image 20250405052437.png](images/Pasted%20image%2020250405052437.png)
![Pasted image 20250405052448.png](images/Pasted%20image%2020250405052448.png)
Here, we defined a function to split the document into smaller chunks. This is important because it has a lot of advantages. One of them includes faster retrieval to improve semantic search precision. Another one includes preserving the context to maintain coherence and easier processing for the model.

The function `chunk_data` accepts these parameters: `docs`, `chunk_size` with a default value of `800`, and `chunk_overlap` with a default value of `100`. `docs` expects the array of `Document` objects we created earlier. `chunk_size` will be the maximum number of characters per chunk while `chunk_overlap` is the number of overlapping characters between consecutive chunks.

Inside `text_splitter`, we used `RecursiveCharacterTextSplitter`. It accepts the `chunk_size` and `chunk_overlap` parameters of the function. It recursively tries different separators (like `\n`, `.`, ` `) to create natural breakpoints. It also ensures that important context is preserved by maintaining a balance between size and readability. We then use the `split_documents()` method to split each `Document` in the `docs` list into smaller chunks while preserving metadata. We then store these new array of `Document` objects in our `doc` variable and return it.

We then called this function and pass in our `doc`, that was created earlier. The `chunk_data()` function processes this list and stores the resulting smaller document pieces in documents. We then printed out the contents of our `documents` variable and print its length.
### Embedding
![Pasted image 20250406033946.png](images/Pasted%20image%2020250406033946.png)
Inside `embeddings`, we initialize an instance of `OpenAIEmbeddings` by passing the OpenAI API key from the environment variable. `OpenAIEmbeddings`, once again, allows you to convert text into vector embeddings. This will be a list of floating-point numbers representing the semantic meaning of the input text.

The `embed_query()` method takes a string input, in this case "Hello world". This will be our test string and we will send it to OpenAI’s API to get an embedding. The result, vectors, is a high-dimensional vector. We measured the length using `len()` and the size of the dimension is `1536`. This vector will be used as input into a vector store for our Pinecone database.
### Vector Search in Pinecone
![Pasted image 20250405053924.png](images/Pasted%20image%2020250405053924.png)
We must first initialize the Pinecone client using the API key, which is pulled from the environment variables using `os.getenv()`. `Pinecone()` here sets up the connection so LangChain can interact with the Pinecone account to query our vector indexes. An index in Pinecone is like a table in a database in the SQL world. This is where all your document vectors will be stored. `"langchain-vectors"` is the name of the index that will be accessed.

`PineconeVectorStore.from_documents()` is a LangChain helper function that takes in a list of `Document` objects to add to the vector store, uses the embeddings model to convert each document into a vector, and stores those vectors inside the Pinecone index named `"langchain-vectors"` (LangChain, 2023b). We pass in `docs` as our `Document` array and `embeddings` as our embeddings model. Then we pass in our `index_name`. This process enables fast semantic retrieval. This means you can now ask natural language questions and Pinecone will return the most relevant documents based on vector similarity.
### Cosine Similarity
![Pasted image 20250406034130.png](images/Pasted%20image%2020250406034130.png)
We will define a `retrieve_query` function to perform a similarity search on the Pinecone index created earlier. `index.similarity_search()` returns the top k most relevant documents based on semantic similarity between the query and document vectors. This allows the system to fetch content that matches the meaning of the query, not just keyword matches.

We will then import some additional modules from LangChain. `load_qa_chain` loads a prebuilt chain that connects the documents to the LLM for answering questions (LangChain, 2023c). `ChatOpenAI` is used to access the GPT-4o model, which is optimized for chat-style outputs and natural language understanding (LangChain, 2025n). 

Here, we're seeing deprecation warnings because `ChatOpenAI` from `langchain_community` is outdated. `langchain_openai` must be used instead. The `stuff` chain is also deprecated, and the error advised us to migrate to modern retrieval-based pipelines using RAG (Retrieval-Augmented Generation).

We are going to initialize GPT-4o in our `llm` variable. We are going to set the `temperature` to `0.6`. Once again, this is just the right amount. Heading over to `chain`, this chain type (`“stuff”`) concatenates all retrieved documents into a single prompt and feeds it to the LLM. The LLM then generates an answer based on the entire context at once.
### Searching the answers
![Pasted image 20250405055629.png](images/Pasted%20image%2020250405055629.png)
This code defines and runs a function called `retrieve_answers`, which performs end-to-end question answering using a retrieval-augmented generation (RAG) pipeline. First, the function takes a user's `query`. It then calls the previously defined `retrieve_query()` function, which searches the Pinecone vector index for documents most relevant to the query using semantic similarity. The returned documents are stored in `doc_search` and this is printed later on. These documents are then passed into a LangChain QA chain using `chain.run()` along with the original question. This chain generates an answer based only on the provided documents. The response generated by the chain is returned to the caller.

The final lines execute the whole process by setting `user_query` to "What are the different models that were used in the stock market?". We then call `retrieve_answers(user_query)`. The result, which is `answer` is printed, and it provides a detailed explanation sourced from the document.
# 5. Google Gemini Pro Demo
In this section, I struggled once again because of the different syntax of old versions from the video and not knowing what to use. But, I settled by using the unified Python SDK, `google-genai`. Let's discuss the difference.

In the video, it uses the older library `google-generativeai`. This is the earlier Python SDK designed for accessing Google's Generative AI models, particularly those developed by Google DeepMind, such as the Gemini series. ​This is deprecated even according to their GitHub repository (Google-Gemini, 2025).

Knowing about this, I decided to use their newer SDK, `google-genai`. This was introduced as a more recent and unified SDK. It aims to provide a consistent interface for Google's Generative AI models across different platforms, including both the Gemini Developer API and Vertex AI. This library supports a broader range of features, including multimodal interactions (text, images, code) and advanced functionalities like code execution, function calling, and integrated Google search grounding (Google, 2025). Here is the link to their [documentation](https://googleapis.github.io/python-genai/).

With all of that out of the way, let's get to the code!
## Setup
![Pasted image 20250405203143.png](images/Pasted%20image%2020250405203143.png)
In the `.env` file, ensure that you have your Google API key. You can get your Google API key by going to [Google AI studio](https://aistudio.google.com/app/apikey). In that page, all you need to do is pressing the Create API Key button.

![Pasted image 20250405203441.png](images/Pasted%20image%2020250405203441.png)
Here are the requirements. We are going to use the `google-genai` SDK. We are also going to use `ipykernel` so that our notebook works. `python-dotenv` will be used as well for the environment variables. Finally, `pillow` will be used for image processing.

![Pasted image 20250405203617.png](images/Pasted%20image%2020250405203617.png)
Here are the imports. We are going to import `textwrap`. ​This is a standard library designed for formatting and wrapping plain text. It provides functions to adjust line breaks, set maximum line widths, and indent text, to enhance readability and presentation (John, 2019). This will be used later on our `to_markdown()` function which will be also explained later.

We are then going to import `genai` from `google` library. This will be important as this will be used for initializing our client. We are also going to import `display` and `Markdown` so that we can display our markdown properly. Lastly, we are going to import `load_dotenv` and `os` once again to import our environment variables.

We are then going to load the environment variables via `load_dotenv()`. We are also going to define a function called `to_markdown()` to convert our text to markdown. It accepts once argument, which is the text itself. It will replace the Unicode bullet points into asterisks so that this will be converted into markdown. Now, we are going to use `textwrap`. Inside the `Markdown()` class, we called the `indent()` function from `textwrap`. This adds a `'>'` character at the beginning of each line in the text. This ensures that this will be displayed in a blockquote. The `predicate` parameter is a function that determines which lines to indent. Here, `lambda _: True` ensures all lines are indented. The `Markdown()` class from the `IPython.display` module renders the indented text as Markdown when we are going to display it.

Now, we are then going to initialize a client object for interacting with Google's Generative AI services using `genai.Client()`. We will pass the API key obtained from the environment variable via `os.getenv()`. The client is authenticated and authorized to make requests to the Generative AI APIs (Google, 2025). This approach ensures that the API key remains confidential and is not exposed in the source code. 
## Code
### Model list
![Pasted image 20250405205327.png](images/Pasted%20image%2020250405205327.png)
In this section, we will list the models according to the search query. There were a lot of models that this code listed, so I just used a condition to filter out the model list to contain only models that were Gemini 2 and above. I defined a `search_query` variable for readability.

Inside the `for` loop, we will iterate through the list of models accessible via the client object.​ `client.models.list()` calls the API to retrieve all available models. This method returns a list of model objects, each containing attributes like `name`, `display_name`, and `description`. We will be only using `name` attribute for this. We will then check if the `search_query` substring exists within the model's `name` attribute. We will then output the name of each model that matches the search criterion by using `print()`. The above image shows the list of models.
### Generating text
![Pasted image 20250405205956.png](images/Pasted%20image%2020250405205956.png)
![Pasted image 20250405210012.png](images/Pasted%20image%2020250405210012.png)
![Pasted image 20250405210027.png](images/Pasted%20image%2020250405210027.png)
In the code block above, as you can see we used a so-called magic command, which is `%%time`. This command measures the execution time of the entire code cell.​ When placed at the beginning of a cell in IPython or Jupyter Notebook, it outputs the CPU and wall time taken to execute the cell (The IPython Development Team, 2025).

We are then going to generate content with `client.models.generate_content`. We are then going to use the `models.generate_content()` method from the initialized `client` before to send a prompt to the specified AI model. We will then retrieve the generated response to our variable `response`. The parameters that we passed are the `model` and the `contents`. For the `model` we used `'gemini-2.5-pro-exp-03-25'`, which is the latest model of Gemini. For the `contents` we passed `'What is the meaning of life?'`, which is the prompt sent to the AI model.

We then formatted the response using `to_markdown()` for better readability. We pass in our `response.text` to get the generated response of the AI. As you can see, the AI responded plentifully.
### Safety rating
![Pasted image 20250405211203.png](images/Pasted%20image%2020250405211203.png)
In this section, we will discuss about the safety rating of the model. In the video, it used `prompt_feedback` to determine the safety rating of the response of the model. It seems that it's not really used anymore and it's probably managed by Google themselves. When we call `prompt_feedback` there is supposed to appear a safety rating but it did not show up. I still included this to show the AI's responses on harmful content.

In this code, we used the same code as the previous section. We tried to request a roast from the AI. The AI returned a message that says it cannot fulfill the request. This shows the behavior of the model on harmful requests.
### Candidates
![Pasted image 20250405211651.png](images/Pasted%20image%2020250405211651.png)
![Pasted image 20250405211707.png](images/Pasted%20image%2020250405211707.png)
![Pasted image 20250405211723.png](images/Pasted%20image%2020250405211723.png)
![Pasted image 20250405211734.png](images/Pasted%20image%2020250405211734.png)
![Pasted image 20250405211750.png](images/Pasted%20image%2020250405211750.png)
The usage of `candidates` is associated with the older version of the Generative AI Python SDK, which has been deprecated. The candidates feature was a feature to represent an alternative response generated by the AI model. Each candidate includes various attributes such as `content`, `finish_reason`, and `safety_ratings`. ​

In this code, I asked the AI "What could be the future of Generative AI?". Again, it responded plentifully once again.
### Streaming
![Pasted image 20250405212337.png](images/Pasted%20image%2020250405212337.png)
![Pasted image 20250405212350.png](images/Pasted%20image%2020250405212350.png)
![Pasted image 20250405212410.png](images/Pasted%20image%2020250405212410.png)
Here, we used streaming. It refers to the process of receiving and processing data incrementally as it's generated, rather than waiting for the entire response to be produced before starting processing. This approach is particularly beneficial when dealing with large volumes of data or when low latency is desired.

We used `models.generate_content_stream()` method to initiate a request to the specified AI model, `'gemini-2.0-flash-001'`, to generate content based on the provided prompt. This method returns a generator that yields chunks of the generated content for real-time processing.​ Inside the `contents` parameter, we passed in our prompt to send to the AI, which is 'How do you think the Philippines could be improved?'. ​

The `for` loop iterates over each chunk yielded by the generator.​ We used `print()` to output the text of the current chunk (`chunk.text`). We then print a separator line for readability using​ `print("_" * 80)`. 

As you can see the output and the response time was quick. Streaming is a very powerful way to generate content quickly. The AI once again answered quite a lot to the question.

The `response_streamed` variable is a generator object, as indicated by the output. Generator objects in Python are iterables that produce items only once and do not store their contents in memory. They are typically used for streaming data or large datasets.

Attempting to access `response_streamed.text` results in an `AttributeError`. This error occurs because generator objects do not have a `text` attribute. To access the content, you must iterate over the generator, as demonstrated in the code.
### Image Processing Using Gemini Flash
![Pasted image 20250405213406.png](images/Pasted%20image%2020250405213406.png)
![Pasted image 20250405213612.png](images/Pasted%20image%2020250405213612.png)
![Pasted image 20250405213628.png](images/Pasted%20image%2020250405213628.png)
![Pasted image 20250405213639.png](images/Pasted%20image%2020250405213639.png)
![Pasted image 20250405213650.png](images/Pasted%20image%2020250405213650.png)
Here, I actually struggled with this bit. It was not accepting the Pillow image at first so I searched for fixes. This is the point where I learned about the new SDK and decided to convert all my code into using the new SDK.

In the code, We imported the `PIL.Image` module so that we can open images and manipulate them. `PIL.Image.open()` opens the specified image file, `'image.png'`, from the `'images'` directory and assigns it to the variable `img`. This operation does not load the image data into memory immediately but prepares it for further processing. ​

We are then going to define our `search_query` again so that we can find models in our model list in an easier way. We set this to `'flash'`. This sets a string to search for models whose names contain `'flash'`. We are going to use the `for` loop again to iterate through the list of models available in the `client`.

We are going to use `models.generate_content()` again to send a request to the specified model, `'gemini-2.0-flash-001'`, with the Pillow image, `img`, and the question "What is in this image?". The model analyzed the image and generated a short textual description as seen above. `to_markdown()` once again formats the generated text into Markdown for better readability.

We are then going to process the same image again and generate a blog post with that image. Here, we used streaming via `models.generate_content_stream()` for faster processing. The `prompt` variable contains our prompt. We set a prompt instructing the model to write a blog post about the image and emphasize the themes of relaxation in nature and the joy of adventures. As seen above, it generated a well-written blog post.

Let's analyze on how this response was generated. We first set a `accumulated_text` variable to an empty string. This will be helpful to accumulate the text generated in the subsequent loop.​ The `for` loop iterates over chunks of content generated by streaming. Each chunk is a part of the model's response to the combined prompt and image.​ We are going to append the text of the current chunk (`chunk.text`) to `accumulated_text`.​ We are then going to use `print()` to print the current chunk's text without adding a newline at the end.​ This can be seen in the non-formatted output above before the Markdown output. We are then going to use `to_markdown()` to format the accumulated text into Markdown for improved readability.
### Another Image Example
![Pasted image 20250405215107.png](images/Pasted%20image%2020250405215107.png)
![another_img.jpg](another_img.jpg)
![Pasted image 20250405215132.png](images/Pasted%20image%2020250405215132.png)
![Pasted image 20250405215147.png](images/Pasted%20image%2020250405215147.png)
Here, I used an image contains a portrait of me. For context, me and my friend went to the Angono petroglyphs. We found the place disappointing and I had the idea of letting the AI write a blog post that would write about how the place was not so good. I wrote the prompt to the AI with the least amount of context possible and see what the AI would write. As you can see, we used the same code as before.

Here, I rotated the image using `rotate()`. The reason I used this is that, somehow Pillow loads the image in landscape mode even though the image is portrait. I also used the `expand` parameter so that it would not have black bars on the side.
### Chat Feature
![Pasted image 20250405215635.png](images/Pasted%20image%2020250405215635.png)
![Pasted image 20250405215649.png](images/Pasted%20image%2020250405215649.png)
​This code demonstrates how to initiate a chat session with Gemini. We demonstrated how to send messages to Gemini and retrieve the chat history. Let's dive deeper on to the code.

We first need to initialize a chat session using `chats.create()`. We are going to store this inside `chat` variable. This creates a new chat session using the specified model, `'gemini-2.5-pro-exp-03-25'`. This session manages the conversation history and facilitates message exchanges with the AI model.​

We then send a message to the model using `chat.send_message()`. We are going to pass our message "Hello" as a parameter. This will send the message to the AI model within the established chat session. The model processes the input and generates a response, which is stored in the response object.​ We are then going to display the response using `to_markdown()`.

We defined a function to retrieve the chat history. This function is `chat_history()`. `chat.get_history()` fetches the list of messages exchanged in the current chat session.​ We are then going to use the `for` loop to iterate through the messages in the history.​ `message_text` retrieves the text of the message, ensuring that if no text is present, "No text" is used as a placeholder.​ `message_role` identifies the sender of the message using `message.role`. This is typically labeled as 'user' or 'model'.​ We are then going to use `to_markdown()` and store this inside `markdown_text`. Inside `to_markdown()`, we are going to format each message in the chat history by using this f-string:  `f'**{message_role.capitalize()}**: {message_text}')`. This creates a Markdown-formatted string that highlights the sender's role and their message. The `capitalize()` method ensures that the role is displayed with an initial uppercase letter.​

We are then going to use `display()` with `markdown_text` inside it. This renders the formatted message in the notebook. `The display()` function is part of the `IPython.display` module. 

We are then going to call this function that we just defined to show our message history with the AI model. This displays each message using `display()`. This can be improved by using one whole message and using `display()` to that instead. As you can see, we only have two messages so far

To demonstrate the chat history capabilities of this AI, we are then going to send another message to the AI using `send_message()`. This time, we will say "I'm doing great!". We will display the response of the AI using `to_markdown()` first to show what the AI responded. We are then going to call `chat_history()` again to see that our new message and response was stored in the history.
# 6. Multi Language Invoice Extractor using Gemini
In this section, we are going to make yet another Streamlit application. This section was an easy one to make so I was able to make it fast. This time we are going to extract and interpret information from invoice images using Google's Gemini AI model once again. The application allows users to upload an image of an invoice and input a text prompt, which is then processed by the AI model to generate a response based on the content of the invoice. The application is set up to handle multiple languages to make it versatile for users from different linguistic backgrounds.
## Setup
![Pasted image 20250406005238.png](images/Pasted%20image%2020250406005238.png)
Again, we are going to put our Gemini API key in our `.env` file. To get an API key, go to [Google AI studio](https://aistudio.google.com/app/apikey) as mentioned previously. I will be using the same API key as from the previous section.

![Pasted image 20250406005913.png](images/Pasted%20image%2020250406005913.png)
Here are the contents for `requirements.txt`. We will be using `streamlit`, `google-genai`, and the LangChain libraries. We are also going to use `python-dotenv` for getting the environment variables and `pillow` for image manipulation. These libraries were already discussed from the previous sections. In the original video, for some reason `chromadb` and `PyPDF2` was included even though they were not used in the code at all. I have no idea what was the reason for the inclusion of such libraries.
## Code
### Library Imports
![Pasted image 20250406010142.png](images/Pasted%20image%2020250406010142.png)
Here, we are going to import the necessary modules. `load_dotenv` will be used to load our environment variables from the `.env` file. `streamlit` will be the main module that we will be using to build our app as well. The `os` module is utilized to access environment variables. The `PIL` library (Python Imaging Library) is imported for image processing, and the `google` module, specifically `genai`, is used to interact with Google's AI models. The `load_dotenv()` function is called to load environment variables from our `.env` file.

![Pasted image 20250406010405.png](images/Pasted%20image%2020250406010405.png)

A client is instantiated using `genai.Client`. The API key is retrieved from the environment variables. The specific AI model used is identified as `'gemini-2.0-flash-001'`.

![Pasted image 20250406010512.png](images/Pasted%20image%2020250406010512.png)

The `get_gemini_response` function is defined to handle the interaction with the AI model. It takes three arguments: `input` (a predefined prompt for the AI model), `image` (the uploaded invoice image), and `prompt` (a user-provided text prompt). The function constructs a `contents` list containing the input text, image data, and prompt, formatted in a way that the AI model can process. All of them have the roles of user and are broken into three different parts. The `models.generate_content()` method is called to generate a response from the AI model, which is then returned as `response.text`.

![Pasted image 20250406010715.png](images/Pasted%20image%2020250406010715.png)

The `input_image_setup` function is designed to process the uploaded image file. It checks if a file has been uploaded, and if so, retrieves its byte data. This data is then structured into a format compatible with the AI model, specifically as an inline data object with a MIME type of "image/jpeg". If no file is uploaded, the function raises a `FileNotFoundError`, ensuring that the application only proceeds when an image is provided.

![Pasted image 20250406010753.png](images/Pasted%20image%2020250406010753.png)

The Streamlit interface is configured with a page title and header, both labeled "Multilanguage Invoice Extractor." User inputs are gathered through a text input field for the prompt and a file uploader for the invoice image. If an image is uploaded, it is displayed on the page with a caption. A button labeled "Tell me about the invoice" triggers the processing of the uploaded image and text prompt. Upon clicking the button, the `input_image_setup()` function is called to prepare the image data, and the `get_gemini_response()` function generates a response from the AI model. We pass `input_prompt` as our system prompt, `image_data` as our image in bytes, and `input` as the user prompt that was taken from the text input box. This response is then displayed on the Streamlit page.
## App
![Pasted image 20250406011220.png](images/Pasted%20image%2020250406011220.png)
This is what the app will look like when we try to run `streamlit run app.py`. As you can see, there is an input prompt with the text box. This is where the users will ask about the uploaded invoice image. There is also a drag and drop component inside the app where the user can upload their invoice images. At the bottom, there is a button, in which when it is clicked, it tells the user about the invoice based on the user prompt and the invoice.

![Pasted image 20250406011705.png](images/Pasted%20image%2020250406011705.png)
Here, we tried to upload an invoice. The image is displayed on the screen when it was uploaded. Let's try asking the AI what the invoice is all about.

![Pasted image 20250406011822.png](images/Pasted%20image%2020250406011822.png)
![Pasted image 20250406011846.png](images/Pasted%20image%2020250406011846.png)

Here, we typed the input prompt to ask the AI what the invoice is all about. It told us the general information about the invoice. It is quite detailed.

![Pasted image 20250406011956.png](images/Pasted%20image%2020250406011956.png)
![Pasted image 20250406012003.png](images/Pasted%20image%2020250406012003.png)

Here is another example. I tried to be more specific with the question and asked the AI who the customer is. It answered correctly.
# 7. Conversational Q&A Chatbot Using Gemini
In this section, we created another Streamlit application that creates a simple chat interface with Gemini once again. The application allows users to input questions or queries, sends these queries to the Gemini model using Google's API, and then displays the AI-generated responses. The chat history is maintained throughout the session and it is displayed in the application.
## Setup
![Pasted image 20250406041734.png](images/Pasted%20image%2020250406041734.png)
Here are our requirements for this application. We have used these libraries before so it is not unfamiliar for us. `streamlit` is for the framework of our application, `google-genai` is for our Gemini API, and `python-dotenv` is for our environment variables.

![Pasted image 20250406041855.png](images/Pasted%20image%2020250406041855.png)

Inside your `.env` file, you will insert you Google API key. Once again, this is important to call the Gemini bot. Let's get to the code now.
## Code
### Imports
![Pasted image 20250406042003.png](images/Pasted%20image%2020250406042003.png)
The code begins by loading environment variables using the `dotenv` library, which is a common practice for managing configuration settings like API keys in a secure manner. The `streamlit` library is then imported to build the web interface, while `os` is used to access environment variables. The `google.genai` module is imported to interact with the Google AI services. The `load_dotenv()` function is called to load environment variables from the `.env` file. 
### Client Setup
![Pasted image 20250406042147.png](images/Pasted%20image%2020250406042147.png)
We are now going to create a client object from the `genai` library using an API key retrieved from the environment variables. This client is used to communicate with the Gemini model which is specified by the string `'gemini-2.5-pro-exp-03-25'` in the `model` variable. We are then going to instantiate a chat session using `client.chats.create()`.
### Function Definition
![Pasted image 20250406042323.png](images/Pasted%20image%2020250406042323.png)
A function named `get_gemini_response` is defined to send a query to the Gemini model and return the response. This function utilizes the `send_message_stream` method of the chat object, which allows for streaming responses. It also accepts the `query` parameter which is the user query.
### Streamlit App Structure
![Pasted image 20250406042420.png](images/Pasted%20image%2020250406042420.png)
![Pasted image 20250406042438.png](images/Pasted%20image%2020250406042438.png)
The Streamlit application is configured with a page title "Q&A Demo with Gemini," and a header "Chat with Gemini" is displayed to the user. The code checks if `'chat_history'` exists in the session state, initializing it as an empty list if not. This list is used to store the conversation history.

An input field is created using `st.text_input` for users to type their queries, and a submit button is provided with `st.button`. When the button is clicked and the input is not empty, the `get_gemini_response()` function is called with the user's input. The response is processed in a loop to accumulate text chunks, which are then displayed in real-time using a placeholder created by `st.empty()`.

The chat history is updated after each interaction and it appends both the user's input and the AI's response. Finally, the chat history is displayed in a subheader section, iterating over each entry and outputting the role (either "You" or "Bot") and the corresponding text message. This allows users to review the entire conversation during their session.
## App
![Pasted image 20250406042846.png](images/Pasted%20image%2020250406042846.png)
Here is what the app looks like if it was ran via `streamlit run app.py`. Let's ask something interesting to the AI. I will ask what are some lovely dishes in Pampanga.

![Pasted image 20250406042953.png](images/Pasted%20image%2020250406042953.png)
![Pasted image 20250406043010.png](images/Pasted%20image%2020250406043010.png)
![Pasted image 20250406043023.png](images/Pasted%20image%2020250406043023.png)

You may not see the live chatting feature in the screenshots, but you have seen another plentiful response from the AI. It answered quite a lot and it even included some desserts which I haven't heard before (they do sound tasty though!). You can also see the chat history between you and the AI.
# References
1. LangChain. (2025a). _Introduction | LangChain_. Retrieved April 3, 2025, from https://python.langchain.com/docs/introduction/
2. OpenAI. (2025a). _GitHub - openai/openai-python: The official Python library for the OpenAI API_. GitHub. Retrieved April 3, 2025, from https://github.com/openai/openai-python
3. HuggingFace. (2025a). _GitHub - huggingface/huggingface_hub: The official Python client for the Huggingface Hub._ GitHub. Retrieved April 3, 2025, from https://github.com/huggingface/huggingface_hub
4. Theskumar. (2025). _GitHub - theskumar/python-dotenv: Reads key-value pairs from a .env file and can set them as environment variables. It helps in developing applications following the 12-factor principles._ GitHub. Retrieved April 3, 2025, from https://github.com/theskumar/python-dotenv
5. Streamlit. (2025). _Streamlit documentation_. Streamlit Documentation. Retrieved April 3, 2025, from https://docs.streamlit.io/
6. LangChain. (2025b). _OpenAI | ️🔗 LangChain_. Retrieved April 3, 2025, from https://python.langchain.com/docs/integrations/providers/openai/#llm
7. IBM, Murel, J., & Noble, J. (2025, January 13). What is LLM temperature? _IBM Think_. Retrieved April 3, 2025, from https://www.ibm.com/think/topics/llm-temperature
8. LangChain. (2025c). _Hugging Face | ️🔗 LangChain_. Retrieved April 3, 2025, from https://python.langchain.com/v0.1/docs/integrations/platforms/huggingface/
9. Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fedus, W., Li, E., Wang, X., Dehghani, M., Brahma, S., Webson, A., Gu, S. S., Dai, Z., Suzgun, M., Chen, X., Chowdhery, A., Narang, S., Mishra, G., Yu, A., . . . Wei, J. (2022). Scaling Instruction-Finetuned Language Models. _arXiv (Cornell University)_. https://doi.org/10.48550/arxiv.2210.11416
10. LangChain. (2025d). _Prompt Templates | ️🔗 LangChain_. Retrieved April 3, 2025, from https://python.langchain.com/docs/concepts/prompt_templates/
11. LangChain. (2025e). _SimpleSequentialChain — 🦜🔗 LangChain  documentation_. Retrieved April 3, 2025, from https://python.langchain.com/api_reference/langchain/chains/langchain.chains.sequential.SimpleSequentialChain.html
12. LangChain. (2022). _Sequential Chains — 🔗 LangChain 0.0.107_. Retrieved April 3, 2025, from https://langchain-doc.readthedocs.io/en/latest/modules/chains/generic/sequential_chains.html
13. LangChain. (2025f). _Messages | ️🔗 LangChain_. Retrieved April 3, 2025, from https://python.langchain.com/docs/concepts/messages
14. LangChain. (2025g). _ChatPromptTemplate — 🦜🔗 LangChain  documentation_. Retrieved April 3, 2025, from https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html
15. LangChain. (2025h). _BaseOutputParser — 🦜🔗 LangChain  documentation_. Retrieved April 3, 2025, from https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.base.BaseOutputParser.html
16. LangChain. (2025i). _How to chain runnables | 🦜️🔗 LangChain_. Retrieved April 3, 2025, from https://python.langchain.com/docs/how_to/sequence/
17. LangChain. (2025j). _Cassandra | ️🔗 LangChain_. Retrieved April 4, 2025, from https://python.langchain.com/docs/integrations/providers/cassandra/
18. LangChain. (2025k). _VectorStoreIndexWrapper — 🦜🔗 LangChain documentation_. Retrieved April 4, 2025, from https://python.langchain.com/api_reference/langchain/indexes/langchain.indexes.vectorstore.VectorStoreIndexWrapper.html#vectorstoreindexwrapper
19. LangChain. (2025l). _OpenAIEMBeddings | ️🔗 LangChain_. Retrieved April 4, 2025, from https://python.langchain.com/docs/integrations/text_embedding/openai/
20. HuggingFace. (2025b). _GitHub - huggingface/datasets: 🤗 The largest hub of ready-to-use datasets for ML models with fast, easy-to-use and efficient data manipulation tools_. GitHub. Retrieved April 4, 2025, from https://github.com/huggingface/datasets
21. OpenAI. (2025b, February 14). _GitHub - openai/tiktoken: tiktoken is a fast BPE tokeniser for use with OpenAI’s models._ GitHub. Retrieved April 4, 2025, from https://github.com/openai/tiktoken
22. Martin.Thoma & mstamy2. (2022, December 31). _PYPDF2_. PyPI. https://pypi.org/project/PyPDF2/
23. Jin, M., Koh, H. Y., Wen, Q., Zambon, D., Alippi, C., Webb, G. I., King, I., & Pan, S. (2024). A survey on graph Neural networks for Time Series: Forecasting, Classification, Imputation, and anomaly Detection. _IEEE Transactions on Pattern Analysis and Machine Intelligence_, _46_(12), 10466–10485. https://doi.org/10.1109/tpami.2024.3443141
24. DataStax. (2025). _About us | DataStax_. Retrieved April 4, 2025, from https://www.datastax.com/company
25. MLJourney. (2024, December 22). _Pinecone Vector Database: Comprehensive guide_. ML Journey. Retrieved April 5, 2025, from https://mljourney.com/pinecone-vector-database-comprehensive-guide
26. Norrid, J. (2025, March 28). GigaOm: Astra DB vs Pinecone in 4 Vector Database Benchmarks. _DataStax_. https://www.datastax.com/blog/astra-db-vs-pinecone-gigaom-performance-study
27. Unstructured-Io. (2025). _GitHub - Unstructured-IO/unstructured: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines._ GitHub. Retrieved April 5, 2025, from https://github.com/Unstructured-IO/unstructured
28. W3Schools. (2025). _W3Schools.com_. Retrieved April 5, 2025, from https://www.w3schools.com/python/pandas/pandas_intro.asp
29. NumPy Developers. (2024). _What is NumPy? — NumPy v2.2 Manual_. NumPy. Retrieved April 5, 2025, from https://numpy.org/doc/2.2/user/whatisnumpy.html
30. IPython development team. (2025). _Jupyter and the future of IPython — IPython_. Ipython. Retrieved April 5, 2025, from https://ipython.org/
31. LangChain. (2025m). _PyPDFDirectoryLoader — 🦜🔗 LangChain  documentation_. Retrieved April 5, 2025, from https://python.langchain.com/api_reference/community/document_loaders/langchain_community.document_loaders.pdf.PyPDFDirectoryLoader.html
32. LangChain. (2023a). _RecursiveCharacterTextSplitter — 🔗 LangChain 0.0.149_. Retrieved April 5, 2025, from https://lagnchain.readthedocs.io/en/stable/modules/indexes/text_splitters/examples/recursive_text_splitter.html
33. LangChain. (2023b). _PineconeVectorStore — 🦜🔗 LangChain  documentation_. Retrieved April 5, 2025, from https://python.langchain.com/api_reference/pinecone/vectorstores/langchain_pinecone.vectorstores.PineconeVectorStore.html
34. Chopra, R., & Sharma, G. D. (2021). Application of Artificial Intelligence in Stock Market Forecasting: A critique, review, and research agenda. _Journal of Risk and Financial Management_, _14_(11), 526. https://doi.org/10.3390/jrfm14110526
35. LangChain. (2023c). _load_qa_chain — 🦜🔗 LangChain  documentation_. Retrieved April 5, 2025, from https://api.python.langchain.com/en/latest/langchain/chains/langchain.chains.question_answering.chain.load_qa_chain.html
36. LangChain. (2025n). _ChatOpenAI | ️🔗 LangChain_. Retrieved April 5, 2025, from https://python.langchain.com/docs/integrations/chat/openai/
37. Google-Gemini. (2025, January 21). _GitHub - Google-Gemini/deprecated-Generative-AI-Python: This SDK is now deprecated, Use the new Unified GenAI SDK._ GitHub. Retrieved April 5, 2025, from https://github.com/google-gemini/deprecated-generative-ai-python
38. Google. (2025, April 1). _Google Gen AI SDK documentation_. Google Gen AI SDK Documentation. Retrieved April 5, 2025, from https://googleapis.github.io/python-genai/
39. John, G. (2019, July 30). _Python text wrapping and filling_. TutorialsPoint. Retrieved April 5, 2025, from https://www.tutorialspoint.com/python-text-wrapping-and-filling
40. The IPython Development Team. (2025, March 8). _Built-in Magic Commands — IPython 9.0.2 documentation_. IPython Documentation. Retrieved April 5, 2025, from https://ipython.readthedocs.io/en/stable/interactive/magics.html