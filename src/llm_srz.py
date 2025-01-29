from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

import os
from dotenv import load_dotenv

load_dotenv()

class NodeLLM:

    def __init__(self):
        self.llm_model = os.getenv("LLM_MODEL")
        self.question_router = self.build_question_router()
        self.answer_normal = self.build_answer_normal()
        self.question_rewriter = self.build_question_rewriter()
        self.rag_chain = self.build_rag_chain()
        self.retrieval_grader = self.build_retrieval_grader()
        self.hallucination_grader = self.build_hallucination_grader()
        self.answer_grader = self.build_answer_grader()

    def build_question_router(self):
        llm = ChatOpenAI(model=self.llm_model, temperature=0)
        question_router_prompt = PromptTemplate(

            template="""
                "You are an expert at routing a user question to either a vectorstore or a normal LLM call.

                - Route to **vectorstore** if the question requires highly specific, detailed, or fact-based information that is stored in a structured, searchable format (e.g., product specifications, technical data, or niche knowledge).
                - Route to **normal LLM** if the question can be answered with general knowledge, reasoning, or broad expertise. If the answer requires subjective interpretation or general knowledge, it should be handled by the LLM.

                Your task is to assess whether the question requires a more precise, structured answer (vectorstore) or whether the LLM can reliably respond with sufficient accuracy based on its general knowledge.

                Return the routing decision as a JSON object with a single key 'datasource'. The value should be either 'normal_llm' or 'vectorstore' based on your decision.

                Question to route: '''{question}'''
            """,

            # template="""You are an expert at routing a user question to a vectorstore or normal LLM call.
            # Use the vectorstore for questions on LLM osram lamps, bulbs, products and specifications.
            # You do not need to be stringent with the keywords in the question related to these topics.
            # Otherwise, use normal LLM call. Give a binary choice 'normal_llm' or 'vectorstore' based on the question.
            # Return the a JSON with a single key 'datasource' and no preamble or explanation.
            # Question to route: '''{question}'''""",
            input_variables=["question"],
        )
        question_router = question_router_prompt | llm | JsonOutputParser()
        return question_router

    def build_answer_normal(self):
        # Normal LLM
        llm = ChatOpenAI(model=self.llm_model, temperature=0)
        prompt = PromptTemplate(
            template="""You are a question-answering system based on your knowledgebase. Respond politely and in a customer-oriented manner.
            If you don't know the answer, refer to the specifics of the question. What exactly is the customer looking for?
            Return a JSON with a single key 'generation' and no preamble or explanation. Be open and talkative.
            Here is the user question: '''{question}'''""",
            input_variables=["question"],
        )
        answer_normal = prompt | llm | StrOutputParser()
        return answer_normal

    def build_question_rewriter(self):
        # Question Re-writer
        llm = ChatOpenAI(model=self.llm_model, temperature=0)
        question_rewriter_prompt = PromptTemplate(
            template="""You are a question re-writer that converts an input question to a better version optimized for vectorstore retrieval.
            Question: '''{question}'''.
            Improved question:""",
            input_variables=["question"],
        )
        question_rewriter = question_rewriter_prompt | llm | StrOutputParser()
        return question_rewriter

    def build_rag_chain(self):
        # Generation (RAG Prompt)
        llm = ChatOpenAI(model=self.llm_model, temperature=0)
        rag_prompt = PromptTemplate(
            template="""
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

            Question: '''{question}'''

            Here is the retrieved document:
            ------------
            Context: {context}
            ------------
            Answer:""",
            input_variables=["question", "context"],
        )
        rag_chain = rag_prompt | llm | StrOutputParser()
        return rag_chain

    def build_retrieval_grader(self):
        # Grading
        llm = ChatOpenAI(model=self.llm_model, temperature=0)
        retrieval_grader_prompt = PromptTemplate(
            template="""You are a grader evaluating the relevance of a retrieved document to a user question.
            Here is the retrieved document:
            ------------
            {document}
            ------------
            Here is the user question: '''{question}'''
            If the document contains keywords or matching product codes that are related to the user's question, rate it as relevant.
            It doesn't need to be a strict test. The goal is to filter out erroneous retrievals.
            Give a binary rating of 'yes' or 'no' to indicate whether the document is relevant to the question.
            Provide the binary rating as JSON with a single key 'score' and without any preamble or explanation.""",
            input_variables=["question", "document"],
        )
        retrieval_grader = retrieval_grader_prompt | llm | JsonOutputParser()
        return retrieval_grader

    def build_hallucination_grader(self):
        llm = ChatOpenAI(model=self.llm_model, temperature=0)
        hallucination_grader_prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is grounded in facts of the document. \n
            Here are the documents:
            ----------
            {documents}
            ----------
            Here is the answer: '''{generation}'''
            Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            Always answer with 'yes'""",
            input_variables=["generation", "documents"],
        )
        hallucination_grader = hallucination_grader_prompt | llm | JsonOutputParser()
        return hallucination_grader
    
    def build_answer_grader(self):

        llm = ChatOpenAI(model=self.llm_model, temperature=0)
        answer_grader_prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is useful to resolve a question.
            Here is the answer:
            -------
            {generation}
            -------
            Here is the question: '''{question}'''
            Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question.
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            Always reply with 'yes'.""",
            input_variables=["generation", "question"],
        )
        answer_grader = answer_grader_prompt | llm | JsonOutputParser()
        return answer_grader