# LangChain 详尽介绍

## 什么是 LangChain？

LangChain 是一个开源的**大语言模型应用开发框架**，旨在帮助开发者更方便地构建基于 LLM（Large Language Model）的复杂应用。它最初由 Harrison Chase 在 2022 年底创建，目前已成为全球最受欢迎的 LLM 应用开发框架之一（GitHub Star 超过 90k+）。

官方定义：

> LangChain is a framework for developing applications powered by language models.

它的核心思想是：**把大语言模型从“单纯的对话接口”变成“可编程的、模块化的、支持外部工具和记忆的智能体”**。

## 为什么需要 LangChain？

原生调用 LLM（如 OpenAI 的 API）只能做到：

- 输入 Prompt → 输出文字

但实际生产级应用往往需要：

- 结合私有文档做问答（RAG）
- 让模型调用外部工具（搜索、计算器、数据库等）
- 拥有长期记忆（记住用户之前的对话）
- 多步推理（Agent + ReAct）
- 链式组合多个 Prompt
- 结构化输出（JSON、Pydantic）
- 评估与监控

LangChain 正是为了解决这些问题而生的框架。

## 核心组件（最新版本 v0.3+ 架构）

| 组件                   | 包名                     | 功能说明                                                     |
| ---------------------- | ------------------------ | ------------------------------------------------------------ |
| langchain-core         | 核心抽象                 | 基础接口（Prompt、Model、OutputParser 等）                   |
| langchain              | 集成层                   | 各种第三方集成（OpenAI、Anthropic、Ollama、Chroma 等）       |
| langchain-community    | 社区集成                 | 社区维护的几百种第三方工具、向量数据库、检索器等             |
| langchain-experimental | 实验性功能               | 一些不稳定但前沿的功能                                       |
| langgraph              | **状态机工作流**（重磅） | 用于构建多 Actor、循环、有状态的 Agent（2024-2025 主流方案） |
| langserve              | 部署工具                 | 把 Chain/Agent 快速变成 REST API                             |
| langsmith              | 可观测性平台（SaaS）     | Tracing、Debug、Eval、Prompt 管理（独立产品，需要注册）      |

## 核心概念（六大模块）

1. **Models（模型层）**

   - LLMs（如 GPT-4、Claude 3、Gemini）
   - Chat Models（对话模型）
   - Embeddings（嵌入模型）

2. **Prompts（提示词管理）**

   - PromptTemplate
   - ChatPromptTemplate
   - FewShotPromptTemplate
   - PipelinePromptTemplate

3. **Chains（链）**

   - 经典顺序链：LLMChain → RetrievalQAChain → ConversationalRetrievalChain
   - 新版推荐使用 LCEL（LangChain Expression Language）

4. **Memory（记忆）**

   - ConversationBufferMemory
   - ConversationSummaryMemory
   - VectorStore-backed Memory
   - Redis/Zep 等持久化记忆

5. **Retrievers & Indexes（检索与向量存储）**

   - VectorStore（Chroma、Pinecone、Weaviate、Qdrant、Milvus 等）
   - Document Loaders（PDF、Word、CSV、HTML、代码等 100+）
   - Text Splitters
   - Retriever（MultiQuery、ParentDocument、SelfQuery 等）

6. **Agents & Tools（智能体与工具）**
   - ReAct Agent
   - OpenAI Functions / Tools
   - 自定义 Tool
   - 使用 langgraph 构建复杂多代理工作流（当前最推荐方式）

## 最常用的开发模式（2025 年最新推荐）

### 1. LCEL（LangChain Expression Language）—— 新一代链式语法

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

chain = (
    ChatPromptTemplate.from_template("你是一个幽默的段子手，给一个关于{topic}的笑话")
    | ChatOpenAI(model="gpt-4o-mini")
    | StrOutputParser()
)

chain.invoke({"topic": "程序员"})
```

````

### 2. RAG（Retrieval-Augmented Generation）经典流程

```python
# 1. 加载文档 → 2. 切块 → 3. 向量化 → 4. 存入向量库 → 5. 检索 + 生成
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=vectorstore.as_retriever()
)
```

### 3. 使用 LangGraph 构建 Agent 工作流（2025 主推）

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# 定义状态 → 节点 → 边 → 编译成可循环执行的图
graph = StateGraph(AgentState)
graph.add_node("research", research_node)
graph.add_node("write", write_node)
...
app = graph.compile(checkpointer=memory)
```

## 生态与合作伙伴（部分）

- 向量数据库：Pinecone、Weaviate、Chroma、Qdrant、Milvus、PGVector
- 本地模型：Ollama、Llama.cpp、vLLM、LM Studio
- 追踪评估：LangSmith（官方）、Helicone、Phoenix、Trulens
- 部署：LangServe、FastAPI、Vercel、Railway

## 学习资源推荐

- 官方文档（强烈推荐）：https://python.langchain.com/docs/
- LangChain 中文文档：https://www.langchain.com.cn/
- LangGraph 官方教程：https://langchain-ai.github.io/langgraph/
- LangSmith：https://smith.langchain.com/
- GitHub：https://github.com/langchain-ai/langchain
- YouTube 官方频道：LangChain
- B 站搜索“LangChain 中文教程”有大量优质视频

## 总结

| 你想做的事             | 推荐方案（2025）                  |
| ---------------------- | --------------------------------- |
| 简单问答               | 直接调用 LLM                      |
| 文档问答、知识库       | LCEL + RAG                        |
| 需要记忆的聊天机器人   | LCEL + RunnableWithMessageHistory |
| 需要调用工具的智能体   | LangGraph + Tools                 |
| 多代理协作、复杂工作流 | LangGraph                         |
| 生产部署               | LangServe + FastAPI               |
| 调试、评估、优化       | LangSmith                         |

LangChain 已经从“链式框架”进化成了一个**完整的 LLM 应用全栈生态**，是目前最成熟、最广泛使用的开发框架之一。

掌握 LangChain ≈ 掌握了 80% 的 LLM 应用开发能力。
````
