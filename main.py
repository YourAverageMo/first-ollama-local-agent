from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from vector import retriever


def main():
	print("Hello from first-ollama-local-agent!")
	model = OllamaLLM(model="llama3.2")

	template = """
    you are an expert in answering questions about a pizza restaurant

    here are some relevant reviews: {reviews}

    here is the question to answer: {question}
    """

	prompt = ChatPromptTemplate.from_template(template)
	chain = prompt | model

	while True:
		print("\n\n----------------------------")
		question = input("Ask your question (q to quit): ")
		print("\n\n")
		if question == "q":
			break

		reviews = retriever.invoke(question)
		result = chain.invoke({"reviews": reviews, "question": question})
		print(result)


if __name__ == "__main__":
	main()
