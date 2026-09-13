"""
End-to-end pipeline: question -> retrieve -> generate -> print.
Run:
  python -m src.answer "Aarogyasri ki em documents kavali?"
  python -m src.answer "What is PM-JAY?"
"""
import sys

from src.retrieval import Retriever
from src.llm import generate_answer


def answer_question(question: str, verbose: bool = True) -> str:
    if verbose:
        print(f"\n❓ Question: {question}\n")
        print("🔍 Retrieving relevant chunks...")

    retriever = Retriever()
    chunks = retriever.search(question, k=4)

    if verbose:
        print(f"   Found {len(chunks)} chunks.")
        for i, c in enumerate(chunks, 1):
            m = c["metadata"]
            print(f"   [{i}] {m['source_file']} p{m['page']} ({m['scheme']})")
        print("\n🤖 Generating Telugu answer via Groq...\n")

    answer = generate_answer(question, chunks)
    return answer


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.answer \"<question>\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    answer = answer_question(question)

    print("=" * 70)
    print(answer)
    print("=" * 70)


if __name__ == "__main__":
    main()
