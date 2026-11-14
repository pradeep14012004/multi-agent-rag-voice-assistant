"""Explicit five-agent pipeline coordinated by a deterministic orchestrator."""
from dataclasses import dataclass, field
from app.vector_store import search
from app.config import DEFAULT_TOP_K
from app.rag import generate_answer

@dataclass
class ConversationMemoryAgent:
    turns: list[dict] = field(default_factory=list)
    def add(self, question: str, answer: str):
        self.turns.append({"question": question, "answer": answer})
        self.turns = self.turns[-10:]
    def context(self) -> str:
        return "\n".join(f"User: {t['question']}\nAssistant: {t['answer'][:500]}" for t in self.turns[-4:])
    def clear(self):
        self.turns.clear()

class QueryUnderstandingAgent:
    def analyze(self, question: str) -> dict:
        q = " ".join(question.strip().split())
        lowered = q.casefold()
        comparative = ["compare", "difference", "versus", " vs ", "similarities", "differentiate"]
        procedural = ["how to", "how do", "steps", "process", "procedure", "method", "workflow"]
        intent = "comparative" if any(w in lowered for w in comparative) else "procedural" if any(w in lowered for w in procedural) else "factual"
        return {"original_query": q, "search_query": q, "intent": intent}

class ClarificationAgent:
    vague = {"explain it", "what about this", "tell me more", "explain that", "how does it work", "why", "what do you mean"}
    def needs_clarification(self, question: str) -> str:
        q = " ".join(question.strip().lower().split())
        if len(q) < 4 or q in self.vague:
            return "Could you be more specific? Mention the concept or document you want me to explain."
        return ""

class RetrievalAgent:
    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K, domain: str = "All domains"):
        return search(query, top_k=top_k, domain=domain)

class ResponseGenerationAgent:
    def respond(self, question: str, hits: list[dict], memory: str = "") -> str:
        return generate_answer(question, hits, memory)

class MultiAgentOrchestrator:
    def __init__(self):
        self.query_agent = QueryUnderstandingAgent()
        self.clarification_agent = ClarificationAgent()
        self.retrieval_agent = RetrievalAgent()
        self.response_agent = ResponseGenerationAgent()
        self.memory_agent = ConversationMemoryAgent()
    def run(self, question: str, top_k: int = DEFAULT_TOP_K, domain: str = "All domains"):
        clarification = self.clarification_agent.needs_clarification(question)
        if clarification:
            return {"clarification": clarification, "intent": "clarification", "hits": [], "answer": ""}
        understood = self.query_agent.analyze(question)
        hits = self.retrieval_agent.retrieve(understood["search_query"], top_k, domain)
        memory = self.memory_agent.context()
        answer = self.response_agent.respond(question, hits, memory)
        self.memory_agent.add(question, answer)
        return {"clarification": "", "intent": understood["intent"], "hits": hits, "answer": answer}
