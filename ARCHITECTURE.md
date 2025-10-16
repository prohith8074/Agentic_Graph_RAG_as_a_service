# 🏗️ Lyzr Challenge RAG System - Complete Architecture & Technical Documentation

## Table of Contents
- [System Overview](#system-overview)
- [Complete Architecture](#complete-architecture)
- [Layered Architecture](#layered-architecture)
- [Data Flow Architecture](#data-flow-architecture)
- [Database Architecture](#database-architecture)
- [Plugin Architecture](#plugin-architecture)
- [AI Model Integration](#ai-model-integration)
- [API & UI Architecture](#api--ui-architecture)
- [Configuration Management](#configuration-management)
- [Monitoring & Debugging](#monitoring--debugging)

## 📋 System Overview

The **Lyzr Challenge RAG System** is a sophisticated **Retrieval-Augmented Generation** platform that processes PDF documents, extracts knowledge graphs, and provides intelligent query responses using multiple AI models and databases.

### Key Features
- **Intelligent Query Routing**: Automatically selects between graph and vector databases based on query intent.
- **Multi-Modal Retrieval**: Combines structured graph queries with semantic vector search.
- **Comprehensive Observability**: Full LLM tracing and performance monitoring.
- **Production-Ready**: Modular architecture with robust error handling.
- **Scalable Design**: Easy to extend and maintain.
- **Pluggable Architecture**: Support for multiple graph databases (Neo4j, Neptune).
- **Version Control**: Ontology versioning with rollback capabilities.
- **REST API**: Complete API interface with streaming support.

## System Architecture

### High-Level Architecture

```mermaid
graph LR
  %% ==== STYLE DEFINITIONS ====
  classDef user fill:#f9f5ff,stroke:#b38cff,stroke-width:2px,color:#3d0073;
  classDef etl fill:#fff8ef,stroke:#f2b279,stroke-width:2px,color:#4b2900;
  classDef db fill:#fff4f4,stroke:#ff9e9e,stroke-width:2px,color:#5a0000;
  classDef agent fill:#edf4ff,stroke:#6da8ff,stroke-width:2px,color:#002d80;
  classDef ret fill:#f2fbf9,stroke:#7dd3b0,stroke-width:1px,color:#003d33;
  classDef obs fill:#fff0f6,stroke:#f2a6c9,stroke-width:1px,color:#660033;

  %% ==== TITLE ====
  subgraph TITLE[" "]
    style TITLE fill:transparent,stroke:transparent
    T["Agentic Graph RAG — Clean Flow"]
    style T fill:transparent,stroke:transparent,color:#000,font-size:24px,font-weight:bold
  end

  %% ==== MAIN PHASES ====
  subgraph INGESTION ["Phase 1: Document Ingestion"]
    direction TD
    UDoc["📄 User Document"]:::user
    ETL1["1. Parse Document"]:::etl
    ETL2["2. Contextualize & Enrich"]:::etl
    ETL3["3. Extract Entities & Relations"]:::etl
    ETL4["4. Build Knowledge Graph"]:::etl
    ETL5["5. Create Text Embeddings"]:::etl

    UDoc --> ETL1 --> ETL2
    ETL2 --> ETL3 --> ETL4
    ETL2 --> ETL5
  end

  subgraph DATABASES ["Phase 2: Unified Data Storage"]
    direction TD
    DBGraph["🕸️ Knowledge Graph DB<br/>(Neo4j)"]:::db
    DBVector["📦 Vector DB<br/>(Qdrant)"]:::db
  end

  subgraph QUERYING ["Phase 3: Query & Response"]
    direction TD
    UQuery["🧠 User Query / Response"]:::user
    Router["⚙️ Query Router"]:::agent
    Redis["⚡ Short-Term Memory<br/>(Redis)"]:::db
    Retrieval["🔍 Retrieval Tools"]:::ret
    Fusion["🔀 Hybrid Scorer"]:::ret
    Synth["🤖 Response Generator"]:::ret

    UQuery -- "Query" --> Router
    Router <--> Redis
    Router --> Retrieval --> Fusion --> Synth
    Synth -- "Final Answer" --> UQuery
  end

  %% ==== OBSERVABILITY & MANAGEMENT ====
  subgraph MANAGEMENT ["Observability & Management"]
      direction TD
      Opik["📊 LLM Tracer<br/>(Opik)"]:::obs
      Ontology["🧩 Ontology Editor<br/>(Gradio)"]:::obs
  end


  %% ==== CONNECTIONS BETWEEN PHASES ====
  ETL4 --> DBGraph
  ETL5 --> DBVector
  Retrieval -.-> DBGraph
  Retrieval -.-> DBVector

  %% ==== MANAGEMENT CONNECTIONS ====
  MANAGEMENT -.-> INGESTION
  MANAGEMENT -.-> QUERYING

end
```

### Component Architecture Breakdown

#### 1. Database Adapters Layer (`database_adapters/`)
```
database_adapters/
├── __init__.py              # Abstract base classes and interfaces
├── neo4j_adapter.py         # Neo4j implementation (Cypher)
├── neptune_adapter.py       # AWS Neptune implementation (Gremlin)
├── database_factory.py      # Factory pattern for adapter creation
└── vector/
    └── qdrant_adapter.py    # Qdrant implementation
```

#### 2. API Layer (`apis/`)
```
apis/
├── __init__.py             # FastAPI app creation
├── routes.py               # REST endpoints with streaming
└── sdk.py                  # Python SDK for programmatic access
```

#### 3. Core Processing Layer (`core/`)
```
core/
├── pdf_parser.py        # Document Ingestion (LlamaParse)
├── node_processor.py    # Text Processing & Normalization
└── contextual_retriever.py # Contextual Enhancement (Cohere)
```

#### 4. Knowledge Graph Layer (`knowledge_graph/`)
```
knowledge_graph/
├── entity_extractor.py  # Entity Extraction (Cohere)
├── graph_builder.py     # Graph Construction & Versioning
└── neo4j_manager.py     # Database Operations
```

#### 5. Query Layer (`query/`)
```
query/
├── query_router.py      # Intelligent Routing (Groq)
├── graph_query.py       # Structured Queries (Cypher)
├── vector_query.py      # Semantic Search (Qdrant)
└── fusion.py            # Hybrid Scoring (RRF)
```

### Data Flow Architecture

#### Ingestion Pipeline
1.  **User Document** → **1. Parse Document** (LlamaParse) → **Parsed Nodes**
2.  **Parsed Nodes** → **2. Contextualize & Enrich** (Cohere) → **Enriched Nodes**
3.  **Enriched Nodes** → **3. Extract Entities & Relations** (Cohere) → **Raw Knowledge Graph**
4.  **Raw Knowledge Graph** → **4. Build Knowledge Graph** → **Versioned Graph**
5.  **Enriched Nodes** → **5. Create Text Embeddings** (Cohere) → **Vector Embeddings**
6.  **Versioned Graph** → **Knowledge Graph DB** (Neo4j)
7.  **Vector Embeddings** → **Vector DB** (Qdrant)

#### Query Pipeline
1.  **User Query** → **Query Router** (Groq) → **Retrieval Plan**
2.  **Retrieval Plan** → **Retrieval Tools** (Neo4j, Qdrant) → **Retrieved Contexts**
3.  **Retrieved Contexts** → **Hybrid Scorer** (RRF) → **Fused Context**
4.  **Fused Context** → **Response Generator** (Cohere) → **Final Answer**
5.  **Final Answer** → **User**

## Technology Stack & Tool Selection

### Core Technologies

| Component | Technology | Why Chosen |
|-----------|------------|------------|
| **Query Routing** | Groq | **Fast inference** for real-time agentic decisions. |
| **Contextual Enhancement** | Cohere | **Excellent summarization** and instruction-following. |
| **Answer Synthesis** | Cohere | **Superior instruction following** for generating high-quality, structured answers. |
| **PDF Parsing** | LlamaParse | **Advanced layout understanding**; handles text, tables, and images in a single API call. |
| **Graph Database** | Neo4j | **Mature ecosystem**, powerful Cypher query language, and excellent visualization tools. |
| **Vector Database** | Qdrant | **High-performance** and supports the advanced metadata filtering needed for logical retrieval. |
| **Embeddings** | Cohere (`embed-english-v3.0`) | **Top-tier performance** for retrieval tasks and seamless integration. |
| **LLM Tracing** | Opik | **LLM-specific features** for monitoring, debugging, and performance tracking. |
| **Short-Term Memory** | Redis | **Microsecond latency** for storing and retrieving conversation history. |
| **UI & Visualization** | Gradio | **Rapid development** of a functional UI for demos and internal tools. |

## Setup & Installation

### Prerequisites
- Python 3.8+
- Git
- Neo4j instance (local or cloud)
- API keys for required services (Cohere, Groq, LlamaParse)

### Clone Repository
```bash
git clone <repository-url>
cd lyzr_challenge
```

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env
```

Required environment variables:
```env
# API Keys
LLAMA_CLOUD_API_KEY=your_llama_cloud_key
COHERE_API_KEY=your_cohere_key
GROQ_API_KEY=your_groq_key

# Neo4j Configuration
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Qdrant Configuration (example for local, in-memory)
QDRANT_LOCATION=":memory:"

# Opik (Optional)
OPIK_API_KEY=your_opik_key
```

## Usage Guide

Please see the `README.md` for a detailed usage guide, including commands for running the Gradio UI and the command-line interface.
