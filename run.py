import os
import asyncio
from AGRAG import AGRAG, QueryParam
from AGRAG.llm import gpt_4o_mini_complete

WORKING_DIR = "./working_dir"

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

async def main():
    rag = AGRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
    )

    # Insert the .txt dataset (for a single file, can be used for test)
    with open("Insert the txt file here", "r", encoding="utf-8") as f:
        content = f.read()

    await rag.ainsert(content)

    # Run the query
    question = ("Insert the question here")

    answer = await rag.aquery(question, param=QueryParam(mode="hybrid"))
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
