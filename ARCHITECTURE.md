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

# 🏗️ Architecture

The system is designed with a modular, multi-layer architecture that separates concerns from data ingestion to user interaction.

## New Features

### Ontology Versioning

The ontology editor now supports versioning. Every time a change is applied to the ontology, a new version is created. This allows users to track the evolution of the ontology and to revert to previous versions if needed.

### Download Graph

Users can now download the current ontology as a JSON file. This is useful for backing up the ontology, for sharing it with others, or for using it in other applications.

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
QDRANT_URL='Qdrant_url'
QDRANT_API_KEY='your_api_key'
QDRANT_COLLECTION_NAME='collection name'
# Opik (Optional)
OPIK_API_KEY=your_opik_key
OPIK_PROJECT_NAME=''
#Redis
REDIS_HOST='your redis host url'
REDIS_PORT='port number'
REDIS_PASSWORD='your_redis_passsword'
```

## Usage Guide

Please see the `README.md` for a detailed usage guide, including commands for running the Gradio UI and the command-line interface.



