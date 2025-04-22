from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template =  """
You are a coding assistant whose task is to generate docstrings for existing Python code.
You will receive code without any docstrings.
Generate the appropiate docstrings for each function, class or method.

Do not return any code. Use the context only to learn about the code.
Write documentation only for the code provided as input code.

The docstring for a function or method should summarize its behavior, side effects, exceptions raised,
and restrictions on when it can be called (all if applicable).
Only mention exceptions if there is at least one _explicitly_ raised or reraised exception inside the function or method.
The docstring prescribes the function or method’s effect as a command, not as a description; e.g. don't write “Returns the pathname ...”.
Do not explain implementation details, do not include information about arguments and return here.
If the docstring is multiline, the first line should be a very short summary, followed by a blank line and a more ellaborate description.
Write single-line docstrings if the function is simple.
The docstring for a class should summarize its behavior and list the public methods (one by line) and instance variables.

In the Argument object, describe each argument. In the return object, describe the returned values of the function, if any.

You will receive a JSON template. Fill the slots marked with <SLOT> with the appropriate description. Return as JSON.

Here is the function to provide the docstring for: {function_code}

Here is the JSON template: {json_template}
"""


prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

json_template = """


"""

function_code = """
"""

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    resume = retriever.invoke(question)
    result = chain.invoke({"function_code": function_code, "json_template" : json_template, "question": question})
    print(result)