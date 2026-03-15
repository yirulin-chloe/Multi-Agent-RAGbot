import os
import re
from typing import List, Dict, Any, Optional, Tuple

# OpenAI (new SDK)
from openai import OpenAI

# Pinecone (v3)
from pinecone import Pinecone


MODEL_NAME = "gpt-4.1-nano"
EMBED_MODEL = "text-embedding-3-small"


def _safe_get_text_from_match(match: Dict[str, Any]) -> str:
    """
    Pinecone match may store original chunk text in metadata. Common keys:
    - "text"
    - "chunk"
    - "content"
    """
    md = match.get("metadata") or {}
    for key in ("text", "chunk", "content"):
        if key in md and isinstance(md[key], str):
            return md[key]
    # fallback: try values
    return ""


class Obnoxious_Agent:
    def __init__(self, client) -> None:
        # TODO: Initialize the client and prompt for the Obnoxious_Agent
        self.client = client
        self.prompt = (
            "You are a strict moderation classifier. You must classify the user's query as either 'Yes' or 'No' ONLY.\n"
            "Yes = query contains any harassing, insults, condescending remarks, "
            "belittling language, hateful, sexually explicit, violent threats, negative attitude or targeted insults.\n"
            "No = query is safe.\n"
            "Do NOT explain your answer. Do NOT refuse. Output EXACTLY one token: Yes or No.\n\n"
            "Examples:\n"
            "User: Hello, how are you?\nAnswer: No\n"
            "User: You are terrible!\nAnswer: Yes\n"
            "User: Go kill yourself\nAnswer: Yes\n"
            "User: What is machine learning?\nAnswer: No\n"
        )
        self.prompt = (
            "You are a strict moderation classifier. You must classify the user's query as either 'Yes' or 'No' ONLY.\n"
            "Yes = query contains harassing, hateful, sexually explicit, violent threats, negative attitude, sarcasm / passive insultor or targeted insults.\n"
            "No = query is safe.\n"
            "Do NOT explain your answer. Output EXACTLY one token: Yes or No.\n\n"
            "Examples:\n"
            "User: Hello, how are you?\nAnswer: No\n"
            "User: You are terrible!\nAnswer: Yes\n"
            "User: Go kill yourself\nAnswer: Yes\n"
            "User: What is machine learning?\nAnswer: No\n"
        )


    def set_prompt(self, prompt):
        # TODO: Set the prompt for the Obnoxious_Agent
        self.prompt = prompt

    def extract_action(self, response) -> bool:
        # TODO: Extract the action from the response
        """
        Return True if obnoxious, False otherwise.
        """
        text = (response or "").strip()
        # robust normalize
        text = re.sub(r"[^A-Za-z]", "", text).lower()
        if text == "yes":
            return True
        if text == "no":
            return False
        return True

    def check_query(self, query):
        # TODO: Check if the query is obnoxious or not
        msg = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": query},
        ]
        resp = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=msg,
            temperature=0,
        )
        out = resp.choices[0].message.content
        return self.extract_action(out)


class Context_Rewriter_Agent:

   def __init__(self, openai_client: OpenAI):
        self.client = openai_client

        self.sys_prompt = (
            "You rewrite the user's latest message into a fully standalone query "
            "using the conversation history.\n\n"

            "If the user uses pronouns like 'it', 'that', 'this', or vague phrases "
            "like 'compare to', 'tell me more', 'explain further', you MUST resolve "
            "what they refer to using the previous conversation.\n\n"

            "Examples:\n"
            "Conversation:\n"
            "User: What is underfitting?\n"
            "User: Compare to overfitting\n"
            "Rewrite: Compare underfitting to overfitting.\n\n"

            "Conversation:\n"
            "User: Explain gradient descent\n"
            "User: How does it work?\n"
            "Rewrite: How does gradient descent work?\n\n"

            "Conversation:\n"
            "User: What is SVM?\n"
            "User: Tell me more\n"
            "Rewrite: Explain SVM in more detail.\n\n"

            "Return ONLY the rewritten query. Do NOT answer."
            )

   def rephrase(self, user_history: List[Dict[str, str]], latest_query: str) -> str:
        if not user_history:
            return latest_query

        history_text = ""
        for item in user_history[-10:]:
            history_text += f"{item.get('role','user')}: {item.get('content','')}\n"

        msg = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": f"Conversation:\n{history_text}\nLatest:\n{latest_query}\n\nRewrite:"},
        ]
        resp = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=msg,
            temperature=0,
        )
        rewritten = (resp.choices[0].message.content or "").strip()
        return rewritten or latest_query



class Query_Agent:
  # TODO: Initialize the Query_Agent agent
   def __init__(self, pinecone_index, openai_client: OpenAI, embeddings=None) -> None:
        self.index = pinecone_index
        self.client = openai_client
        self.embeddings = embeddings
        self.prompt = (
            "You are a router for a domain-specific assistant.\n"
            "The assistant answers questions about machine learning.\n\n"

            "Your task is to classify the user's query into ONE of the following categories:\n\n"

            "1. Relevant → The query contains ANY question about machine learning, "
            "even if it also includes unrelated content or small talk.\n\n"

            "2. Small_Talk → ONLY very short greetings and "
            "these MUST NOT contain a question or topic.\n"
            "Examples: hello, hi, good morning, how are you, thanks, goodbye.\n"
            "These are short conversational phrases only.\n\n"

            "3. Irrelevant → Any question or topic unrelated to machine learning.\n"
            "Examples: \n"
            "Do you prefer coffee or tea?\n"
            "What is your favorite recipe?\n"
            "Who won the Super Bowl?\n"
            "Tell me a joke.\n"

            "IMPORTANT RULE:\n"
            "If the query contains at least one machine learning question, "
            "it MUST be classified as Relevant, even if other unrelated topics are included.\n\n"

            "Return EXACTLY one word:\n"
            "Relevant\n"
            "Small_Talk\n"
            "Irrelevant\n"
            "No extra text."
        )

   def _embed(self, text: str) -> List[float]:
        emb = self.client.embeddings.create(
            model=EMBED_MODEL,
            input=text,
        )
        return emb.data[0].embedding

   def query_vector_store(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        vec = self._embed(query)
        res = self.index.query(
            vector=vec,
            top_k=k,
            include_metadata=True,
            namespace="ns2500",  # or ns1000, ns2500
        )

        matches = res.get("matches", []) if isinstance(res, dict) else getattr(res, "matches", [])
        docs = []
        for m in matches or []:
            docs.append(
                {
                    "id": m.get("id"),
                    "score": m.get("score"),
                    "metadata": m.get("metadata", {}),
                    "text": _safe_get_text_from_match(m),
                }
            )
        return docs

   def set_prompt(self, prompt: str):
        self.prompt = prompt

   def extract_topic(self, response: str) -> str:
       text = (response or "").strip()
       text = re.sub(r"[^A-Za-z_]", "", text)

       if text == "Relevant":
           return "Relevant"
       if text == "Irrelevant":
           return "Irrelevant"
       if text == "Small_Talk":
           return "Small_Talk"

       # safer fallback
       return "Irrelevant"


   def is_relevant_topic(self, query: str) -> str:
        msg = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": query},
        ]
        resp = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=msg,
            temperature=0,
        )
        out = resp.choices[0].message.content
        return self.extract_topic(out)


class Clean_Query_Agent:
    def __init__(self, openai_client):
        self.client = openai_client

        self.clean_prompt = (
            "You are a strict filter for a machine learning question answering system.\n\n"
            "The user query may contain multiple questions.\n"
            "Extract ONLY the part related to machine learning.\n"
            "Remove unrelated topics.\n"
            "Remove insults or abusive language.\n\n"
            "If multiple parts exist, keep ONLY the machine learning question.\n"
            "If NO machine learning content exists, output exactly: NONE\n\n"
            "Do NOT answer the question.\n"
            "Return ONLY the cleaned machine learning query.\n\n"
            "Examples:\n"
            "Input: How to cook egg? and what is underfitting?\n"
            "Output: What is underfitting?\n\n"
            "Input: Explain SVM and who won the Super Bowl?\n"
            "Output: Explain SVM.\n\n"
            "Input: You are stupid.\n"
            "Output: NONE\n"
            "Return ONLY the cleaned machine learning query text.\n"
            "Do NOT include the word 'Output'.\n"
        )

    def clean_query(self, query: str) -> str:
        msg = [
            {"role": "system", "content": self.clean_prompt},
            {"role": "user", "content": query},
        ]

        resp = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=msg,
            temperature=0,
        )

        cleaned = resp.choices[0].message.content.strip()
        return cleaned

class LLM_Only_Agent:
    def __init__(self, openai_client: OpenAI):
        self.client = openai_client
        self.sys_prompt = "You are a friendly, helpful assistant. Respond naturally to user input."

    def generate_response(self, query: str, conv_history: List[Dict[str, str]] = None) -> str:
        # Build conversation history if available
        history_text = ""
        for m in (conv_history or [])[-8:]:
            history_text += f"{m.get('role','user')}: {m.get('content','')}\n"

        user_msg = f"Conversation (recent):\n{history_text}\nUser question:\n{query}\nAnswer:"

        resp = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.5,
        )
        return (resp.choices[0].message.content or "").strip()


class Answering_Agent:
    def __init__(self, openai_client: OpenAI) -> None:
        self.client = openai_client
        self.sys_prompt = (
            "You are a helpful assistant. Use the provided context snippets to answer.\n"
            "Cite sources using (Page X) ONLY when page numbers are available.\n"
            "If page numbers are not provided, do not create it yourself"
            "Do not mention snippet numbers.\n"
            "Be concise and correct."
        )


    def generate_response(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        conv_history: List[Dict[str, str]],
        k: int = 5
    ) -> str:
        docs = (docs or [])[:k]
        context_blocks = []

        for i, d in enumerate(docs, start=1):
            text = (d.get("text") or "").strip()
            page = d.get("metadata", {}).get("page_number", None)

            if text:
                if page is not None:
                    context_blocks.append(f"[Page {page}] {text}")
                else:
                    context_blocks.append(f"[Snippet {i}] {text}")

        context = "\n\n".join(context_blocks) if context_blocks else "(No context retrieved.)"


        history_text = ""
        for m in (conv_history or [])[-8:]:
            history_text += f"{m.get('role','user')}: {m.get('content','')}\n"

        user_msg = (
            f"Conversation (recent):\n{history_text}\n\n"
            f"Context:\n{context}\n\n"
            f"User question:\n{query}\n\nAnswer:"
        )

        resp = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": self.sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()


class Relevant_Documents_Agent:
    """
    Restriction: Cannot use LangChain API.
    This agent checks whether retrieved docs are relevant to the query.
    Return "Relevant" / "Irrelevant".
    """


    def __init__(self, openai_client: OpenAI) -> None:
        # TODO: Initialize the Relevant_Documents_Agent
        self.client = openai_client
        self.sys_prompt = (
            "You are a strict relevance judge. Given a user query and retrieved snippets, "
            "decide if the snippets are relevant enough to answer the query.\n"
            "Return ONLY one word: Relevant or Irrelevant.\n"
            "No extra text."
        )

    def get_relevance(self, conversation) -> str:
        # TODO: Get if the returned documents are relevant
        """
        conversation can be a dict with:
          - query: str
          - docs: list[str] or list[dict with 'text']
        """
        query = conversation.get("query", "")
        docs = conversation.get("docs", []) or []
        snippets = []
        for d in docs[:5]:
            if isinstance(d, dict):
                t = (d.get("text") or "").strip()
            else:
                t = str(d).strip()
            if t:
                snippets.append(t[:800])

        joined = "\n\n".join([f"Snippet{i+1}: {s}" for i, s in enumerate(snippets)]) or "(No snippets.)"

        msg = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": f"Query: {query}\n\nRetrieved:\n{joined}\n\nDecision:"},
        ]
        resp = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=msg,
            temperature=0,
        )
        out = (resp.choices[0].message.content or "").strip().lower()
        out = re.sub(r"[^a-z]", "", out)
        if out == "relevant":
            return "Relevant"
        if out == "irrelevant":
            return "Irrelevant"
        return "Irrelevant"


class Head_Agent:
    """
    Controller that manages sub-agents.
    Suggested flow:
      1) Rewrite query (optional)
      2) Obnoxious check
      3) Query agent decides in-domain
      4) Retrieve docs from Pinecone
      5) Relevant-doc check
      6) Answer
    """

    def __init__(self, openai_key, pinecone_key, pinecone_index_name) -> None:
        # TODO: Initialize the Head_Agent
        self.openai_key = openai_key
        self.pinecone_key = pinecone_key
        self.pinecone_index_name = pinecone_index_name

        # Clients
        self.openai_client = OpenAI(api_key=self.openai_key)
        self.pc = Pinecone(api_key=self.pinecone_key)
        self.index = self.pc.Index(self.pinecone_index_name)

        self.setup_sub_agents()

        # Store conversation history in-memory
        self.conv_history: List[Dict[str, str]] = []

    def setup_sub_agents(self):
        # TODO: Setup the sub-agents
        self.obnoxious_agent = Obnoxious_Agent(self.openai_client)
        self.context_rewriter = Context_Rewriter_Agent(self.openai_client)
        self.query_agent = Query_Agent(self.index, self.openai_client, embeddings=None)
        self.clean_query_agent = Clean_Query_Agent(self.openai_client)
        self.relevance_agent = Relevant_Documents_Agent(self.openai_client)
        self.answering_agent = Answering_Agent(self.openai_client)
        self.llm_only_agent = LLM_Only_Agent(self.openai_client)

    def _refusal_irrelevant(self) -> str:
        return "This is not related to machine learning content."

    def _refusal_obnoxious(self) -> str:
        return "Refuse to answer, detected obnoxious content"

    def handle_one_turn(self, user_query: str):

        # 0) rewrite for multi-turn
        rewritten = self.context_rewriter.rephrase(self.conv_history, user_query)

        # 1) parallel checks
        is_bad = self.obnoxious_agent.check_query(rewritten)
        topic_type = self.query_agent.is_relevant_topic(rewritten)

        # 2) routing

        if topic_type == "Irrelevant":
            self.last_agent_path = "REFUSAL_IRRELEVANT"
            return {
                "response": self._refusal_irrelevant(),
                "agent_path": self.last_agent_path
            }

        elif topic_type == "Small_Talk":

            if is_bad:
                self.last_agent_path = "REFUSAL_OBNOXIOUS"
                return {
                    "response": self._refusal_obnoxious(),
                    "agent_path": self.last_agent_path
                }

            self.last_agent_path = "LLM_ONLY"
            resp = self.llm_only_agent.generate_response(rewritten)

            return {
                "response": resp,
                "agent_path": self.last_agent_path
            }

        # 3) relevant query → clean it
        elif topic_type == "Relevant":

            cleaned = self.clean_query_agent.clean_query(rewritten)

            if cleaned is None or cleaned.strip().upper() == "NONE":
                self.last_agent_path = "REFUSAL_IRRELEVANT"
                return {
                    "response": self._refusal_irrelevant(),
                    "agent_path": self.last_agent_path
                }

            rewritten = cleaned
        else:
            raise ValueError(f"Unknown topic type: {topic_type}")

        # 4) retrieve docs
        docs = self.query_agent.query_vector_store(rewritten, k=5)

        # 5) check doc relevance
        rel = self.relevance_agent.get_relevance({
            "query": rewritten,
            "docs": docs
        })

        if rel != "Relevant":

            self.last_agent_path = "LLM_ONLY_FALLBACK"

            resp = self.llm_only_agent.generate_response(
                rewritten,
                conv_history=self.conv_history)

            # update history
            self.conv_history.append({"role": "user", "content": user_query})
            self.conv_history.append({"role": "assistant", "content": resp})

            return {
                "response": resp,
                "agent_path": self.last_agent_path
            }
        else:
          # 6) final RAG answer
          self.last_agent_path = "RETRIEVAL"

          resp = self.answering_agent.generate_response(
              query=rewritten,
              docs=docs,
              conv_history=self.conv_history,
              k=5,
          )

        # update history
        self.conv_history.append({"role": "user", "content": user_query})
        self.conv_history.append({"role": "assistant", "content": resp})

        return {
            "response": resp,
            "agent_path": self.last_agent_path
        }