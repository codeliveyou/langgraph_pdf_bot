import logging
from .state import GraphState
from .llm_srz import NodeLLM
from .utils import initialize_retrievers

nodellm = NodeLLM()
ensemble_retriever = None

def initialize_retriever(doc_splits, embeddings):
    global ensemble_retriever
    ensemble_retriever = initialize_retrievers(doc_splits, embeddings)
    return ensemble_retriever

class GraphNodes:

    @staticmethod
    def retrieve(state: GraphState):
        """Retrieve documents for the question."""
        logging.info("🗄️ Retrieving documents...")
        question = state["question"]
        documents = ensemble_retriever.invoke(question)
        logging.info(f"Retrieved {len(documents)} documents")
        print(f"Retrieved {len(documents)} documents")
        return {"documents": documents, "question": question}

    @staticmethod
    def generate(state: GraphState):
        """Generate an answer using retrieved documents."""
        logging.info("🤖 Generating answer...")
        question = state["question"]
        documents = state["documents"]
        generation = nodellm.rag_chain.invoke({"context": documents, "question": question})
        logging.info(f"Generated answer: {generation[:20]}")
        return {"documents": documents, "question": question, "generation": generation}

    @staticmethod
    def grade_documents(state: GraphState):
        """Grade the relevance of retrieved documents."""
        logging.info("💎 Grading documents...")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = [
            d
            for d in documents
            if nodellm.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )["score"]
            == "yes"
        ]
        logging.info(f"Filtered {len(documents) - len(filtered_docs)} documents")
        return {"documents": filtered_docs, "question": question}

    @staticmethod
    def transform_query(state: GraphState):
        """Re-write the query to improve retrieval."""
        logging.info("📝  Transforming the query...")
        question = state["question"]
        better_question = nodellm.question_rewriter.invoke({"question": question})
        logging.info(f"Improved question: {better_question}")
        return {"documents": state["documents"], "question": better_question}

    @staticmethod
    def normal_llm(state: GraphState):
        logging.info("💭  Calling normal LLM...")
        question = state["question"]
        answer = nodellm.answer_normal.invoke({"question": question})
        logging.info(f"Answer: {answer[:20]}")
        return {"question": question, "generation": answer}

    @staticmethod
    def route_question(state: GraphState):
        """Route the question to either vectorstore or normal LLM."""
        logging.info("⚖️  Routing the question...")
        question = state["question"]
        source = nodellm.question_router.invoke({"question": question})
        logging.info(f"Routing to: {source}")
        return "normal_llm" if source["datasource"] == "normal_llm" else "vectorstore"

    @staticmethod
    def decide_to_generate(state: GraphState):
        """Decide whether to generate or rephrase the query."""
        logging.info("🗯️  Deciding to generate or rephrase the query...")
        return "transform_query" if not state["documents"] else "generate"

    @staticmethod
    def grade_generation(state: GraphState):
        """Grade the generation and its relevance."""
        logging.info("🔍 Grading the generation...")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]
        hallucination_score = nodellm.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )["score"]
        logging.info(f"Grounded in the documents: {hallucination_score}")
        return "useful" if hallucination_score == "yes" else "not supported"