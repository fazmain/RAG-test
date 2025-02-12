# qa_retrieval.py

import os
from dotenv import load_dotenv

# Load environment variables from .env if available.
load_dotenv()

# Import updated modules from langchain_community.
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def main():
    # Retrieve API keys and environment variables.
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    # Here we use "PINECONE_ENV"; ensure your .env uses the same key.
    pinecone_region = os.environ.get("PINECONE_ENV")

    # Ensure all required variables are set.
    if not all([openai_api_key, pinecone_api_key, pinecone_region]):
        raise ValueError("Missing API keys or environment variables. "
                         "Set OPENAI_API_KEY, PINECONE_API_KEY, and PINECONE_ENV.")

    # Initialize the OpenAI embeddings module with your API key.
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Load the existing Pinecone vector store.
    # Make sure that the index name here ("test-2") matches the one used when embedding.
    index_name = "test-2"
    vectorstore = Pinecone.from_existing_index(index_name, embeddings)

    # Create a retriever from the vector store.
    # This retriever will perform a similarity search and return the top 4 most relevant documents.
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Initialize the OpenAI LLM with a specified temperature.
    llm = OpenAI(temperature=0.8, openai_api_key=openai_api_key)

    # Build the RetrievalQA chain using the "stuff" chain type.
    # The "stuff" chain type concatenates all the retrieved documents into one context.
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # --- Inspecting the RAG Pipeline ---

    # Access and print the prompt template used to combine the documents with the question.
    # In the current version, the prompt template is stored inside the llm_chain of the combine_documents_chain.
    prompt_template = qa.combine_documents_chain.llm_chain.prompt.template
    print("=== Prompt Template ===")
    print(prompt_template)
    print("=======================\n")

    # Start an interactive query loop.
    query = input("Enter your question (or type 'exit' to quit): ")

    while query.lower() != "exit":
        # --- Step 1: Retrieve Relevant Documents ---
        # Retrieve the top matching documents (context) for the query.
        docs = retriever.get_relevant_documents(query)
        print("\n=== Retrieved Documents (Context) ===")
        for i, doc in enumerate(docs, start=1):
            print(f"Document {i}:\n{doc.page_content}\n")
        print("=====================================\n")

        # --- Step 2: Construct the Final Prompt ---
        # Combine the content of all retrieved documents into one context string.
        context = "\n".join([doc.page_content for doc in docs])
        # Format the final prompt using the prompt template.
        # This is the prompt that will be sent to the LLM.
        final_prompt = qa.combine_documents_chain.llm_chain.prompt.format(
            question=query, context=context
        )
        print("=== Final Prompt Sent to LLM ===")
        print(final_prompt)
        print("================================\n")

        # --- Step 3: Generate the Answer ---
        # Run the QA chain with the query to generate an answer.
        answer = qa.run(query)
        print("=== Answer ===")
        print(answer)
        print("==============\n")

        # Prompt the user for another query.
        query = input("Enter your question (or type 'exit' to quit): ")

if __name__ == "__main__":
    main()
