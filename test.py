from langchain.llms import OpenAI

import constants

llm = OpenAI(openai_api_key=constants.APIKEY)

llm_result = llm("What you can do for me")

print(llm_result)