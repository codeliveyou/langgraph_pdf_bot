from langgraph.graph import END, START, StateGraph
from .state import GraphState
from .nodes import GraphNodes as nodes



class GraphWorkflow:

    def __init__(self):
        self.app = self.build_graph()

    def build_graph(self):

        workflow = StateGraph(GraphState)
        workflow.add_node("normal_llm", nodes.normal_llm)
        workflow.add_node("retrieve", nodes.retrieve)
        workflow.add_node("grade_documents", nodes.grade_documents)
        workflow.add_node("generate", nodes.generate)
        workflow.add_node("transform_query", nodes.transform_query)

        workflow.add_conditional_edges(
            START, nodes.route_question, {"normal_llm": "normal_llm", "vectorstore": "retrieve"}
        )
        workflow.add_edge("normal_llm", END)
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            nodes.decide_to_generate,
            {"transform_query": "transform_query", "generate": "generate"},
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate", nodes.grade_generation, {"not supported": "generate", "useful": END}
        )

        return workflow.compile()