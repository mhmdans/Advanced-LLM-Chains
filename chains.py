from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableBranch
from dotenv import load_dotenv
import os

load_dotenv()

llm_api = os.getenv("OPENROUTER_API_key")

llm = ChatOpenAI(
    openai_api_base="https://openrouter.ai/api/v1",
    openai_api_key=llm_api,
    model="qwen/qwen2.5-vl-32b-instruct:free"
)

sequence_chain = RunnableSequence([
    lambda x: f"Generate a short summary for: {x}",
    llm,
    lambda x: f"Summary result: {x.content if hasattr(x, 'content') else str(x)}"
])

parallel_chain = RunnableParallel({
    "formal": lambda x: llm.invoke(f"Rewrite formally: {x}"),
    "friendly": lambda x: llm.invoke(f"Rewrite in a friendly tone: {x}")
})

branch_chain = RunnableBranch(
    branches=[
        (lambda x: "error" in x.lower(), lambda x: llm.invoke(f"Explain this error clearly: {x}")),
        (lambda x: "summarize" in x.lower(), lambda x: llm.invoke(f"Summarize this content: {x}")),
    ],
    default=lambda x: llm.invoke(f"Answer concisely: {x}")
)

if __name__ == "__main__":
    user_input = "Explain subnetting in simple terms."
    print("\n=== Sequence Chain ===")
    seq_output = sequence_chain.invoke(user_input)
    print(seq_output)

    print("\n=== Parallel Chain ===")
    par_output = parallel_chain.invoke(user_input)
    print("Formal:", par_output["formal"].content)
    print("Friendly:", par_output["friendly"].content)

    print("\n=== Branch Chain ===")
    branch_output = branch_chain.invoke("Summarize how routers forward packets.")
    print(branch_output.content)
