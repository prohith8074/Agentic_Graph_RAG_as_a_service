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
- [Recent Fixes & Updates](#recent-fixes--updates)

## 📋 System Overview

The **Lyzr Challenge RAG System** is a sophisticated **Retrieval-Augmented Generation** platform that processes PDF documents, extracts knowledge graphs, and provides intelligent query responses using multiple AI models and databases.

### Key Features
- **Intelligent Query Routing**: Automatically selects between graph and vector databases based on query intent
- **Multi-Modal Retrieval**: Combines structured graph queries with semantic vector search
- **Comprehensive Observability**: Full LLM tracing and performance monitoring
- **Production-Ready**: Modular architecture with robust error handling
- **Scalable Design**: Easy to extend and maintain
- **Pluggable Architecture**: Support for multiple graph databases (Neo4j, Neptune)
- **Version Control**: Ontology versioning with rollback capabilities
- **REST API**: Complete API interface with streaming support

## 🔍 Recent Fixes & Database Issues Resolved

### **✅ Issues Fixed:**

1. **Neo4j Embedding Storage Error** ❌ → ✅ **FIXED**
   - **Problem**: `"Collections containing collections can not be stored in properties"`
   - **Root Cause**: Entity embeddings were nested arrays that Neo4j cannot store
   - **Solution**: Enhanced property filtering in `Neo4jAdapter` and `Neo4jManager` to exclude complex fields

2. **Cohere API Response Format** ❌ → ✅ **FIXED**
   - **Problem**: Cohere embeddings returned inconsistent structure format
   - **Solution**: Added proper handling for both `response.embeddings.float_` and `response.embeddings` formats

3. **Duplicate Configuration** ❌ → ✅ **FIXED**
   - **Problem**: `QDRANT_DB_PATH` defined twice in `settings.py`
   - **Solution**: Removed duplicate definition

4. **Missing QDRANT Configuration** ❌ → ✅ **ADDED**
   - **Problem**: QDRANT variables used but not defined in environment
   - **Solution**: Added QDRANT configuration to `.env` template

### **Database Connection Issues Identified:**

1. **Environment Variables**
   - Missing QDRANT_URL and QDRANT_API_KEY in `.env` file
   - Duplicate QDRANT_DB_PATH definitions in settings

2. **Cohere API Response Handling**
   - Inconsistent handling of embedding response structure
   - Fixed in: `utils/embeddings.py`, `utils/memory_manager.py`, `main.py`

3. **Neo4j Property Filtering**
   - Complex nested structures (embeddings) not properly filtered
   - Fixed in: `database_adapters/neo4j_adapter.py`, `knowledge_graph/neo4j_manager.py`

4. **GraphBuilder Embedding Management**
   - Embeddings were being added to entities before Neo4j storage
   - Fixed by separating embedding generation from graph storage

### **Key Files Modified:**
- `lyzr_challenge/database_adapters/neo4j_adapter.py` - Enhanced property filtering
- `lyzr_challenge/knowledge_graph/neo4j_manager.py` - Consistent property filtering
- `lyzr_challenge/knowledge_graph/graph_builder.py` - Separated embeddings from graph storage
- `lyzr_challenge/utils/embeddings.py` - Fixed Cohere response handling
- `lyzr_challenge/utils/memory_manager.py` - Fixed embedding format handling
- `lyzr_challenge/main.py` - Fixed Cohere embedding access
- `lyzr_challenge/config/settings.py` - Removed duplicate configuration

### **Testing:**
- Created `test_neo4j_fix.py` for validation
- All database connection and population issues resolved
- System now properly handles complex data structures

## System Architecture

### Updated High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              User Interfaces                                       │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                      │
│  │   CLI Tool      │ │   REST API      │ │   Python SDK    │                      │
│  │                 │ │   (FastAPI)     │ │                 │                      │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘                      │
└─────────────────────────┬───────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────────────────────┐
│                              Query Router (Groq LLM)                              │
│           ┌─────────────────────────────────────────────────────────────────┐      │
│           │ Advanced Reasoning: Multi-step + Streaming + Hybrid Scoring   │      │
│           └─────────────────────────────────────────────────────────────────┘      │
└─────────────────────────┬───────────────────────────────────────────────────────────┘
                    ┌─────┴─────┐
          ┌─────────▼──┐ ┌─────▼─────────┐ ┌─────────────────────────────────────┐
          │ Graph DB   │ │ Vector DB     │ │         Memory Systems             │
          │ (Neo4j/    │ │ (ChromaDB)    │ │  ┌─────────────┬─────────────────┐  │
          │  Neptune)  │ │               │ │  │ Short-term │ Long-term       │  │
          │            │ │               │ │  │ (Redis)    │ (ChromaDB)      │  │
          └────────────┘ └───────────────┘ │  └─────────────┴─────────────────┘  │
                    ▲             ▲         └─────────────────────────────────────┘
                    │             │
          ┌─────────┴─────────────┴─────────────────────────────────────────────────┐
          │                   Knowledge Processing Pipeline                        │
          │  ┌─────────────┬─────────────┬─────────────┬─────────────┐            │
          │  │ PDF Parser  │ Contextual  │ Entity      │ Ontology    │            │
          │  │ (LlamaParse)│ Retrieval  │ Extraction  │ Editor      │            │
          │  │             │ (Cohere)   │ (Cohere)    │ (Versioned)  │            │
          │  └─────────────┴─────────────┴─────────────┴─────────────┘            │
          └────────────────────────────────────────────────────────────────────────┘
                                          ▲
                                          │
                            ┌─────────────┴─────────────┐
                            │      Data Sources         │
                            │    PDF Documents         │
                            └──────────────────────────┘
```

### Component Architecture Breakdown

#### 1. Database Adapters Layer (`database_adapters/`)
```
database_adapters/
├── __init__.py              # Abstract base classes and interfaces
├── neo4j_adapter.py         # Neo4j implementation (Cypher)
├── neptune_adapter.py       # AWS Neptune implementation (Gremlin)
├── database_factory.py      # Factory pattern for adapter creation
└── README.md               # Adapter documentation
```

#### 2. API Layer (`apis/`)
```
apis/
├── __init__.py             # FastAPI app creation
├── routes.py               # REST endpoints with streaming
├── sdk.py                  # Python SDK for programmatic access
└── run_api.py             # Server runner script
```

#### 3. Ontology Versioning Layer (`ontology_versioning/`)
```
ontology_versioning/
├── __init__.py            # Version control implementation
└── README.md              # Versioning documentation
```

#### 4. Testing Layer (`tests/`)
```
tests/
├── __init__.py            # Test configuration and fixtures
├── test_database_adapters.py  # Database adapter tests
├── test_query_system.py   # Query system tests
├── test_apis.py          # API endpoint tests
└── test_versioning.py    # Ontology versioning tests
```

#### 5. Configuration Layer (`config/`)
```
config/
├── settings.py          # Environment-based configuration
│   ├── API Keys Management
│   ├── Model Selection
│   ├── Database Connections
│   └── Logging Configuration
└── __init__.py
```

#### 2. Core Processing Layer (`core/`)
```
core/
├── pdf_parser.py        # Document Ingestion
│   ├── LlamaParse Integration
│   ├── Document Chunking
│   └── Metadata Extraction
├── node_processor.py    # Text Processing
│   ├── Node Validation
│   ├── Text Normalization
│   └── Content Extraction
├── contextual_retriever.py # Contextual Enhancement
│   ├── Cohere LLM Integration
│   ├── Sliding Window Context
│   └── Summary Generation
└── __init__.py
```

#### 3. Knowledge Graph Layer (`knowledge_graph/`)
```
knowledge_graph/
├── entity_extractor.py  # Entity Extraction
│   ├── Cohere LLM for NER
│   ├── Relationship Mining
│   └── JSON Parsing & Validation
├── graph_builder.py     # Graph Construction
│   ├── Entity Deduplication
│   ├── Relationship Merging
│   └── Graph Validation
├── neo4j_manager.py     # Database Operations
│   ├── Connection Management
│   ├── Schema Creation
│   ├── Data Population
│   └── Query Execution
└── __init__.py
```

#### 4. Query Layer (`query/`)
```
query/
├── query_router.py      # Intelligent Routing
│   ├── Groq LLM for Intent Analysis
│   ├── Routing Decision Logic
│   └── Fallback Mechanisms
├── graph_query.py       # Structured Queries
│   ├── Cypher Query Generation
│   ├── Neo4j Execution
│   └── Result Synthesis
├── vector_query.py      # Semantic Search
│   ├── ChromaDB Integration
│   ├── Embedding Search
│   └── Context Retrieval
└── __init__.py
```

#### 5. Utilities Layer (`utils/`)
```
utils/
├── helpers.py           # Common Utilities
│   ├── Logging Setup
│   ├── Data Validation
│   └── Performance Monitoring
├── opik_tracer.py       # Observability
│   ├── LLM Call Tracking
│   ├── Performance Metrics
│   └── Error Logging
└── __init__.py
```

### Data Flow Architecture

#### Ingestion Pipeline
1. **PDF Document** → LlamaParse → **Structured Documents**
2. **Documents** → Node Processing → **Text Chunks**
3. **Chunks** → Contextual Retrieval → **Enhanced Chunks**
4. **Enhanced Chunks** → Entity Extraction → **Knowledge Graphs**
5. **Knowledge Graphs** → Graph Builder → **Unified Graph**
6. **Unified Graph** → Neo4j → **Persistent Storage**

#### Query Pipeline
1. **User Query** → Intent Analysis → **Routing Decision**
2. **Graph Route** → Cypher Generation → Neo4j Query → **Structured Results**
3. **Vector Route** → Semantic Search → Context Retrieval → **Relevant Chunks**
4. **Results** → LLM Synthesis → **Final Answer**

## Technology Stack & Tool Selection

### Core Technologies

#### Large Language Models (LLMs)

| Component | Technology | Alternative Options | Why Chosen |
|-----------|------------|-------------------|------------|
| **Query Routing** | Groq (gpt-oss-20b) | OpenAI GPT-4, Anthropic Claude, Google Gemini | **Fast inference** (optimized for speed), **Cost-effective** for intent classification, **Reliable API** with good JSON parsing |
| **Contextual Enhancement** | Cohere Command A | OpenAI GPT-3.5, Anthropic Claude | **Excellent summarization** capabilities, **Rate-limited friendly** pricing, **Consistent output quality** |
| **Answer Synthesis** | Cohere Command A | Same as above | **Superior instruction following**, **Good for structured tasks**, **Cost-effective** for synthesis tasks |

#### Document Processing

| Component | Technology | Alternatives | Why Chosen |
|-----------|------------|-------------|------------|
| **PDF Parsing** | LlamaParse | PyPDF2, PDFMiner, Docling, Unstructured.io | **Advanced layout understanding**, **Table/chart extraction**, **Metadata preservation**, **Handles complex PDFs** |
| **Text Chunking** | LlamaIndex | LangChain, Custom implementations | **Production-ready**, **Multiple strategies**, **Metadata handling**, **Integration ecosystem** |

#### Databases & Storage

| Component | Technology | Alternatives | Why Chosen |
|-----------|------------|-------------|------------|
| **Graph Database** | Neo4j | Amazon Neptune, JanusGraph, ArangoDB | **Mature ecosystem**, **Cypher query language**, **Excellent visualization**, **ACID compliance** |
| **Vector Database** | ChromaDB | Pinecone, Weaviate, Qdrant, FAISS | **Open-source**, **Easy setup**, **Good Python integration**, **Persistent storage**, **No vendor lock-in** |
| **Embeddings** | Nomic Embed | OpenAI Ada, Cohere Embed, Sentence Transformers | **High-quality embeddings**, **Cost-effective**, **Good performance**, **Open-source friendly** |

#### Observability & Monitoring

| Component | Technology | Alternatives | Why Chosen |
|-----------|------------|-------------|------------|
| **LLM Tracing** | Opik | LangSmith, Weights & Biases, Custom logging | **LLM-specific features**, **Easy integration**, **Good visualization**, **Open-source option** |

### Document Processing & Parsing

#### LlamaParse for Document Processing
**Selected Tool**: LlamaParse
**Why Chosen**: Single API call for text chunks + image summaries

**Problem Solved**: Traditional parsing approaches have significant drawbacks:

**Traditional Methods (PymuPDF, unstructured.io, etc.)**:
- Multiple processing steps required
- Separate handling for text and images
- OCR needed for scanned documents
- Complex chunking strategies (recursive text splitter, sentence/paragraph/heading level)
- Multiple API calls for image processing
- Time-consuming and expensive

**LlamaParse Advantages**:
- **Single API Call**: Processes entire document including images in one request
- **LLM-Powered**: Uses advanced language models for understanding document structure
- **Image Summarization**: Automatically extracts and summarizes images
- **Better Accuracy**: Context-aware chunking with metadata preservation
- **Handles Complex PDFs**: Tables, figures, complex layouts

### Contextual Retrieval Implementation

#### Cohere for Contextual Enhancement
**Selected Tool**: Cohere Command A
**Why Chosen**: Unlimited context length with excellent summarization

**Contextual Retrieval Problem**: Standard RAG fails in production (~67% accuracy) due to lack of context.

**Solution Implemented**:
1. **Sliding Window Approach**: 2048 character context window around each chunk
2. **Cohere LLM**: Generates contextual summaries for each chunk
3. **Context Preservation**: Maintains document-level understanding

**Why Cohere**:
- **Unlimited Context**: Can handle large context windows without token limits
- **Cost-Effective**: Reasonable pricing for high-volume processing
- **Quality**: Excellent summarization and contextual understanding
- **Rate Limits**: Suitable for batch processing (10 calls/min on free tier)

**Alternative Considered**: Groq models - good for small documents but hit token limits on large documents.

### Memory Systems Architecture

#### Redis for Short-Term Memory
**Purpose**: Conversation context, recent interactions
- **Fast Access**: In-memory storage for quick retrieval
- **TTL Support**: Automatic expiration of old context
- **Session Management**: User-specific conversation history

#### ChromaDB for Long-Term Memory
**Purpose**: Persistent knowledge storage, user preferences
- **Free & Local**: Perfect for development
- **Embedding Storage**: Vector-based long-term knowledge
- **Scalable**: Can handle large knowledge bases

### Observability & Monitoring

#### Opik for LLM Tracing
**Selected Tool**: Opik (Comet)
**Why Chosen**: Specialized LLM observability with easy integration

**Available Alternatives**:
- **MLflow**: General MLOps, less LLM-specific
- **LangSmith**: Excellent LLM tracing but more complex
- **RAGAS**: Evaluation-focused, less comprehensive tracing
- **Custom Logging**: Would require significant development

**Opik Advantages**:
- **LLM-Specific**: Designed for language model observability
- **Easy Integration**: Simple decorators and automatic tracing
- **Performance Monitoring**: Tracks latency, token usage, costs
- **Error Tracking**: Comprehensive error logging and debugging
- **Open-Source Option**: Cost-effective for development

### Selection Criteria Summary

#### Performance & Speed
- **Groq**: Sub-second response times for routing decisions
- **ChromaDB**: Fast vector similarity search
- **Redis**: Microsecond memory access
- **Neo4j/Neptune**: Optimized graph traversals

#### Cost-Effectiveness
- **Cohere**: Competitive pricing for LLM tasks
- **Nomic**: Free/open-source embeddings
- **ChromaDB**: No per-query costs
- **LlamaParse**: Single API call reduces costs
- **Opik**: Open-source observability

#### Reliability & Production-Readiness
- **Neo4j**: Battle-tested graph database
- **LlamaParse**: Production-ready document processing
- **FastAPI**: Enterprise-grade API framework
- **Streaming Support**: Real-time response capabilities

#### Developer Experience & Extensibility
- **Modular Architecture**: Easy to swap components
- **Factory Pattern**: Pluggable database adapters
- **Comprehensive APIs**: REST + Python SDK
- **Version Control**: Ontology versioning system
- **Testing Infrastructure**: Full CI/CD setup

## Setup & Installation

### Prerequisites
- Python 3.8+
- Git
- Neo4j instance (local or cloud)
- API keys for required services

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
NOMIC_API_KEY=your_nomic_key

# Neo4j Configuration
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# File Paths
PDF_FILE_PATH=/path/to/your/document.pdf

# Opik (Optional)
OPIK_API_KEY=your_opik_key
```

### Database Setup
```bash
# Start Neo4j (if local)
# Or ensure cloud instance is running

# The system will automatically create required constraints
```

## Usage Guide

### Basic Usage
```bash
# Run full pipeline (PDF processing + query interface)
python main.py

# Query-only mode (skip PDF processing)
python main.py --query-only
```

### Interactive Query Interface
```
=== Lyzr RAG Query Interface ===
Ask questions about the processed document.
Type 'exit' to quit, 'stats' for system statistics.

Your question: What are the components of the Transformer architecture?
🤔 Thinking...

============================================================
Query: What are the components of the Transformer architecture?
Routing: graph (confidence: 0.95)
Reasoning: Question about architecture components
Answer: The Transformer architecture consists of an encoder and decoder...
============================================================
```

### Advanced Usage

#### Custom PDF Processing
```python
from core.pdf_parser import PDFParser
from knowledge_graph.entity_extractor import EntityExtractor

# Initialize components
parser = PDFParser()
extractor = EntityExtractor()

# Process custom PDF
documents = parser.parse_pdf("custom.pdf")
# ... rest of pipeline
```

#### Direct Query Routing
```python
from query.query_router import QueryRouter

router = QueryRouter()
result = router.route_query("Explain attention mechanism")
print(result['answer'])
```

## Module Documentation

### Configuration Module (`config/`)

#### `settings.py`
Centralized configuration management using environment variables.

**Key Classes:**
- `Settings`: Main configuration class with validation

**Key Features:**
- Environment variable loading
- Configuration validation
- Default value handling

### Core Processing Module (`core/`)

#### `pdf_parser.py`
Handles PDF document ingestion and initial processing.

**Key Classes:**
- `PDFParser`: Main PDF processing class

**Key Methods:**
- `parse_pdf()`: Parse PDF using LlamaParse
- `extract_nodes()`: Convert documents to nodes

#### `contextual_retriever.py`
Generates contextual summaries for text chunks.

**Key Classes:**
- `ContextualRetriever`: Cohere-based summarization

**Key Methods:**
- `get_sliding_window_context()`: Create context windows
- `generate_contextual_summary()`: LLM-based summarization

### Knowledge Graph Module (`knowledge_graph/`)

#### `entity_extractor.py`
Extracts entities and relationships from text.

**Key Classes:**
- `EntityExtractor`: LLM-based extraction

**Key Methods:**
- `extract_entities_and_relationships()`: Main extraction method
- `process_chunks_batch()`: Batch processing

#### `neo4j_manager.py`
Handles Neo4j database operations.

**Key Classes:**
- `Neo4jManager`: Database interface

**Key Methods:**
- `connect()`: Establish database connection
- `populate_graph()`: Insert knowledge graph
- `execute_query()`: Run Cypher queries

### Query Module (`query/`)

#### `query_router.py`
Intelligent routing between query types.

**Key Classes:**
- `QueryRouter`: Main routing logic

**Key Methods:**
- `analyze_query_intent()`: LLM-based intent analysis
- `route_query()`: Route and execute queries

#### `graph_query.py`
Handles structured graph-based queries.

**Key Classes:**
- `GraphQueryInterface`: Neo4j query interface

#### `vector_query.py`
Handles semantic vector-based queries.

**Key Classes:**
- `VectorQueryInterface`: ChromaDB query interface

### Utilities Module (`utils/`)

#### `helpers.py`
Common utility functions and logging setup.

#### `opik_tracer.py`
Observability and tracing integration.

## API Reference

### Main Classes

#### LyzrRAGSystem
Main orchestrator class.

**Methods:**
- `run_full_pipeline(pdf_path)`: Execute complete processing pipeline
- `interactive_query_loop()`: Start interactive query interface

#### QueryRouter
Intelligent query routing.

**Methods:**
- `route_query(query)`: Route and execute query
- `analyze_query_intent(query)`: Analyze query intent

### Configuration

#### Settings
Configuration management.

**Attributes:**
- API keys, model selections, file paths
- Database connection parameters
- Logging and performance settings

## Troubleshooting

### Common Issues

#### Neo4j Connection Issues
```bash
# Check Neo4j is running
# Verify NEO4J_URI in .env
# Check firewall settings
```

#### API Key Errors
```bash
# Verify all required API keys in .env
# Check API key validity
# Ensure sufficient credits/quotas
```

#### Memory Issues
```bash
# Reduce CONTEXT_WINDOW_SIZE
# Process smaller PDFs
# Increase system memory
```

#### Import Errors
```bash
# Ensure all dependencies installed
pip install -r requirements.txt
# Check Python version compatibility
```

### Performance Optimization

#### For Large Documents
- Increase `MAX_RETRIES` for API calls
- Use smaller `CONTEXT_WINDOW_SIZE`
- Implement batch processing

#### For High Load
- Increase `RATE_LIMIT_DELAY`
- Use connection pooling
- Implement caching layers

### Logging and Debugging

All components include comprehensive logging. Check:
- `logs/lyzr_challenge_YYYYMMDD.log`
- Console output with appropriate log levels
- Opik dashboard for LLM call traces

### Support

For issues not covered here:
1. Check the logs for detailed error messages
2. Verify all prerequisites are met
3. Ensure network connectivity to external services
4. Review API rate limits and quotas

---

*This architecture documentation provides a comprehensive guide to understanding, deploying, and extending the Lyzr Challenge RAG system.*

Parsing:
For parsing the user given document we have number of ways. we can use python modules like Pymupdf,unstructured.io etc.. first we extract the text and convert into chunks using different chunk strategies like Reursivetextcharacter text splitter and sentence level, paragraph level, heading level and then exract the images and use vision capable LLm to summarie the image. To do this entire process it takes much time and need API calls. If the document has scanned pages then we nned to OCR technique or LLM to extract the and repeat the same process and get the text chunks and image summaries. It needs the routing whethere to choose pymupdf/unstructured.io- It takes too much of time.
To overcome these I choose Llama parse:
Why: 
which uses the LLM to parse the document and give the chunks of the document along the image summaries in a single api call . This method also takes time and it costs as it uses LLm but more better than previous method in terms of accuracy. 

Contextual chunks:
Next RAG fails in production 67%- solution is Contextual retrieval. ---Read the Anthropic article

for each chunks we have created (from llama parse) , we get the contextual chunk from the document:
1. By giving whole document along with the chunk for every chunk - results (chunk+contextual chunk)
2. we can give the nearest chunks to get the contextual chunk - 20 chunks 
Both have pros and cons. for a large document method 1 ends up with token limit as we are giving the whole document along with the each chunk. soltuion is to use the prompt Caching freature in LLM calls.

I have tired with Groq models all worked well for small document , but for large document I ended up with the token limit early.
I have used "Cohere" LLM with the second method by giving the 2048  context_window_lenght before and after the chunk to get the contextual chunk for each chunk.- Resukt it worked well. (But the only problem is the api limit 10 call/min since i used free tier & 1000 api calls/month)
Why cohere: it has unlimited context for a limited API . So I choose this LLM.

for RAG:
after getting contextual chunks we need to store it a vector database (chromadb). here I used Nomic Emebeddings which outpermed many embedding models and it is free to use. 

for Graph RAG:
we have contextual chunks for each chunk. Now we need to extract the knowledge(entities and relationships) from the each chunk (Contextual chunks+chunk). Here you can use any LLM which is capable of performing this task. I used the cohere latest model (command-a-vision-2025 model) which is a multimodel LLM specialise in text tasks. 
Next merge the knowledge and perform the deduplication and store it in the Neo4j DB

Querying:
User asks the query and based on the query the Agent will query the vector or Graph or both DBs to answer the query.

Memory:
For short term memory I have used the redis and for long term memory used chromadb
Why redis: Its is open source, and high speed  in retreving the data - to main the long term conversation.
Why chormadb: Its free  and helpful for development locally
For monitoring/tracing:
There are many tool available like MLops, Langsmith,RAGAS etc..
I choose opik(comet).
Why opik: 

For reducing the Database queries use tool result caching, it returns the response if it already answered. There by it cuts off the steps from retrieving the context - for RAG and generating the query and then searching the DB and finally response generation is reduced. Saved the API calls too for repeated queries.  
