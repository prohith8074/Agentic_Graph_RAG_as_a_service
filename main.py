#!/usr/bin/env python3
"""
Main orchestrator script for the Lyzr Challenge RAG system.
Coordinates the entire pipeline from PDF parsing to query answering.
""" 

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, AsyncGenerator
import os
import sys
import atexit
from config.settings import settings
from core.pdf_parser import PDFParser
from core.node_processor import NodeProcessor
from core.contextual_retriever import ContextualRetriever
from knowledge_graph.entity_extractor import EntityExtractor
from knowledge_graph.graph_builder import GraphBuilder
from knowledge_graph.neo4j_manager import Neo4jManager
from query.query_router import QueryRouter
from query.advanced_router import AdvancedQueryRouter
from knowledge_graph.ontology_editor import OntologyEditor

from utils.helpers import setup_logging, format_query_result, get_system_info
from utils.opik_tracer import opik_tracer
from database_adapters.database_factory import init_database_connections, shutdown_database_connections, get_qdrant_client, ensure_qdrant_collection, db_manager

logger = logging.getLogger(__name__)

class LyzrRAGSystem:
    """Main RAG system orchestrator."""

    def __init__(self):
        """Initialize the RAG system."""
        self.pdf_parser = PDFParser()
        self.node_processor = NodeProcessor()
        self.contextual_retriever = ContextualRetriever()
        self.entity_extractor = EntityExtractor()
        self.graph_builder = GraphBuilder()

        # Use global database instances instead of creating new ones
        self.neo4j_adapter = db_manager.get_adapter("neo4j")
        self.qdrant_client = get_qdrant_client()

        # Initialize query routers with global instances
        self.query_router = QueryRouter(neo4j_adapter=self.neo4j_adapter, qdrant_client=self.qdrant_client)
        self.advanced_router = AdvancedQueryRouter(neo4j_adapter=self.neo4j_adapter, qdrant_client=self.qdrant_client)
        self.ontology_editor = OntologyEditor()

        # Pipeline state
        self.documents = None
        self.nodes = None
        self.contextualized_chunks = None
        self.knowledge_graph = None

    def load_and_parse_pdf(self, pdf_path: str) -> bool:
        """
        Load and parse PDF document.

        Args:
            pdf_path: Path to PDF file

        Returns:
            True if successful
        """
        logger.info("=== Step 1: PDF Parsing ===")

        try:
            # Parse PDF
            self.documents = self.pdf_parser.parse_pdf(pdf_path)
            if not self.documents:
                raise ValueError("No documents parsed from PDF")

            # Extract nodes
            self.nodes = self.pdf_parser.extract_nodes(self.documents)
            if not self.nodes:
                raise ValueError("No nodes extracted from documents")

            # Validate nodes
            if not self.node_processor.validate_nodes(self.nodes):
                raise ValueError("Node validation failed")

            logger.info(f"✅ PDF parsing complete: {len(self.documents)} documents, {len(self.nodes)} nodes")
            return True

        except Exception as e:
            logger.error(f"❌ PDF perarsing failed: {e}")
            return False

    async def contextualize_chunks(self) -> bool:
        """
        Generate contextual summaries for chunks.

        Returns:
            True if successful
        """
        logger.info("=== Step 2: Contextual Retrieval ===")

        try:
            if not self.nodes:
                raise ValueError("No nodes available for contextualization")

            self.contextualized_chunks = await self.contextual_retriever.contextualize_chunks(self.nodes)

            if not self.contextualized_chunks:
                raise ValueError("No contextualized chunks generated")

            # Save contextualized chunks for potential use in populate_vector_store.py
            import json
            with open('contextualized_chunks.json', 'w') as f:
                json.dump(self.contextualized_chunks, f, indent=2)
            logger.info("Contextualized chunks saved to contextualized_chunks.json")

            logger.info(f"✅ Contextualization complete: {len(self.contextualized_chunks)} chunks")
            return True

        except Exception as e:
            logger.error(f"❌ Contextualization failed: {e}")
            return False

    async def extract_knowledge_graph(self) -> bool:
        """
        Extract entities and relationships from contextualized chunks.

        Returns:
            True if successful
        """
        logger.info("=== Step 3: Knowledge Graph Extraction ===")

        try:
            if not self.contextualized_chunks:
                raise ValueError("No contextualized chunks available")

            # Prepare text chunks for extraction
            chunk_texts = [chunk['contextualized_text'] for chunk in self.contextualized_chunks]

            # Extract knowledge graphs from chunks
            extracted_graphs = await self.entity_extractor.process_chunks_batch(chunk_texts)

            if not extracted_graphs:
                raise ValueError("No knowledge graphs extracted")

            # Merge and deduplicate
            self.knowledge_graph = self.graph_builder.merge_and_deduplicate_graphs(extracted_graphs)

            # Validate graph structure
            if not self.graph_builder.validate_graph_structure(self.knowledge_graph):
                raise ValueError("Knowledge graph validation failed")

            # Get statistics
            stats = self.graph_builder.get_graph_statistics(self.knowledge_graph)
            logger.info(f"✅ Knowledge graph complete: {stats}")

            return True

        except Exception as e:
            logger.error(f"❌ Knowledge graph extraction failed: {e}")
            return False

    def populate_neo4j(self) -> bool:
        """
        Populate Neo4j with the knowledge graph.

        Returns:
            True if successful
        """
        logger.info("=== Step 4: Neo4j Population ===")

        try:
            if not self.knowledge_graph:
                raise ValueError("No knowledge graph available")

            # Use global database manager to get Neo4j adapter
            neo4j_adapter = db_manager.get_adapter("neo4j")
            if not neo4j_adapter:
                raise ConnectionError("Neo4j adapter not available from global database manager")

            # Create constraints
            if not neo4j_adapter.create_constraints():
                logger.warning("Could not create database constraints")

            # Clear existing data (optional)
            # neo4j_adapter.clear_database()

            # Populate with knowledge graph
            if not neo4j_adapter.populate_graph(self.knowledge_graph):
                raise RuntimeError("Failed to populate Neo4j with knowledge graph")

            logger.info("✅ Neo4j population complete")
            return True

        except Exception as e:
            logger.error(f"❌ Neo4j population failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def validate_chunk_for_embedding(self, chunk: Dict[str, Any], chunk_index: int) -> tuple[bool, str]:
        """
        Validate a chunk to ensure it can be embedded.

        Args:
            chunk: The chunk to validate
            chunk_index: Index of the chunk for logging

        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        try:
            # Check if chunk exists and is a dict
            if chunk is None:
                return False, f"chunk {chunk_index} is None"

            if not isinstance(chunk, dict):
                return False, f"chunk {chunk_index} is not a dictionary (type: {type(chunk)})"

            # Check for required key
            if 'contextualized_text' not in chunk:
                return False, f"chunk {chunk_index} missing 'contextualized_text' key"

            contextualized_text = chunk.get('contextualized_text')

            # Check if text exists
            if contextualized_text is None:
                return False, f"chunk {chunk_index} 'contextualized_text' is None"

            # Check if text is a string
            if not isinstance(contextualized_text, str):
                return False, f"chunk {chunk_index} 'contextualized_text' is not a string (type: {type(contextualized_text)})"

            # Check if text is empty or only whitespace
            stripped_text = contextualized_text.strip()
            if not stripped_text:
                return False, f"chunk {chunk_index} 'contextualized_text' is empty or only whitespace"

            # Check minimum length (avoid very short texts that might not embed well)
            if len(stripped_text) < 10:
                return False, f"chunk {chunk_index} 'contextualized_text' too short ({len(stripped_text)} chars, minimum 10)"

            # Check for excessive special characters (might indicate corrupted text)
            alpha_ratio = sum(c.isalpha() for c in stripped_text) / len(stripped_text)
            if alpha_ratio < 0.1:  # Less than 10% alphabetic characters
                return False, f"chunk {chunk_index} 'contextualized_text' contains too many non-alphabetic characters (alpha ratio: {alpha_ratio:.2f})"

            # Check for binary/unprintable characters
            if any(ord(c) < 32 and c not in '\n\t\r' for c in stripped_text):
                return False, f"chunk {chunk_index} 'contextualized_text' contains unprintable characters"

            return True, ""

        except Exception as e:
            return False, f"chunk {chunk_index} validation error: {str(e)}"

    def filter_chunks_for_embedding(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter out chunks that cannot be embedded.

        Args:
            chunks: List of chunks to filter

        Returns:
            List of valid chunks for embedding
        """
        if not chunks:
            logger.warning("No chunks provided for filtering")
            return []

        valid_chunks = []
        invalid_count = 0

        for i, chunk in enumerate(chunks):
            is_valid, reason = self.validate_chunk_for_embedding(chunk, i + 1)
            if is_valid:
                valid_chunks.append(chunk)
            else:
                logger.warning(f"⚠️ Filtering out invalid chunk {i+1}: {reason}")
                invalid_count += 1

        logger.info(f"✅ Chunk filtering complete: {len(valid_chunks)} valid chunks, {invalid_count} filtered out")
        return valid_chunks

    def populate_vector_store(self) -> bool:
        """
        Populate Qdrant vector store with contextualized chunks using the default embedder.
        This method is now robust, centralized, and includes verification.

        Returns:
            True if successful, False otherwise
        """
        logger.info("=== Step 5: Vector Store Population ===")
        from qdrant_client.models import PointStruct
        from utils.embeddings import default_embedder

        try:
            if not self.contextualized_chunks:
                raise ValueError("No contextualized chunks available for vector storage.")

            logger.info(f"Received {len(self.contextualized_chunks)} chunks for vector store population.")

            # 1. Filter out invalid chunks
            valid_chunks = self.filter_chunks_for_embedding(self.contextualized_chunks)
            if not valid_chunks:
                logger.error("❌ No valid chunks remaining after filtering.")
                return False

            # 2. Prepare documents and metadata for embedding
            docs_to_embed = [chunk['contextualized_text'] for chunk in valid_chunks]
            payloads = [{"document": chunk['original_text'], "id": chunk.get('id')} for chunk in valid_chunks]

            # 3. Generate embeddings using the default embedder
            logger.info(f"Generating embeddings for {len(docs_to_embed)} documents...")
            embeddings = default_embedder.embed_texts(docs_to_embed)
            if not embeddings or len(embeddings) != len(docs_to_embed):
                raise RuntimeError("Embedding generation failed or returned incorrect number of vectors.")

            # 4. Prepare points for Qdrant
            points = [
                PointStruct(id=hash(doc) % (2**63), vector=embedding, payload=payload)
                for doc, embedding, payload in zip(docs_to_embed, embeddings, payloads)
                if embedding is not None and any(v != 0 for v in embedding) # Ensure embedding is not null or all zeros
            ]

            if not points:
                logger.error("❌ No valid points were generated for Qdrant upsert.")
                return False

            # 5. Upsert points to Qdrant
            logger.info(f"Attempting to upsert {len(points)} points to Qdrant...")
            if not ensure_qdrant_collection(settings.QDRANT_COLLECTION_NAME):
                raise ConnectionError(f"Failed to ensure Qdrant collection '{settings.QDRANT_COLLECTION_NAME}' exists.")

            self.qdrant_client.upsert(
                collection_name=settings.QDRANT_COLLECTION_NAME,
                points=points,
                wait=True  # Wait for the operation to complete
            )
            logger.info(f"✅ Upsert operation completed for {len(points)} points.")

            # 6. Verify the operation
            time.sleep(1) # Give Qdrant a moment to index
            collection_info = self.qdrant_client.get_collection(settings.QDRANT_COLLECTION_NAME)
            if collection_info.points_count >= len(points):
                logger.info(f"✅ Verified: {collection_info.points_count} vectors successfully stored in Qdrant!")
                return True
            else:
                logger.warning(f"⚠️ Verification failed: Expected at least {len(points)} points, but found {collection_info.points_count}.")
                return False

        except Exception as e:
            logger.error(f"❌ Vector store population failed: {e}", exc_info=True)
            return False

    def initialize_query_systems(self) -> bool:
        """
        Initialize query systems (vector store should be populated separately).

        Returns:
            True if successful
        """
        logger.info("=== Step 5: Query System Initialization ===")

        try:
            # Query router is initialized in __init__, just verify connections
            stats = self.query_router.get_routing_stats()

            if not stats.get('neo4j_connected', False):
                logger.warning("Neo4j not connected - graph queries will fail")

            if stats.get('vector_stats', {}).get('total_vectors', 0) == 0:
                logger.warning("No vectors in collection - vector queries may fail")

            logger.info(f"✅ Query systems initialized: {stats}")
            return True

        except Exception as e:
            logger.error(f"❌ Query system initialization failed: {e}")
            return False

    async def run_full_pipeline(self, pdf_path: str) -> AsyncGenerator[str, None]:
        """
        Run the complete RAG pipeline, yielding status updates along the way.
        """
        logger.info("🚀 Starting Lyzr RAG Pipeline")
        start_time = time.time()

        try:
            # Step 1: PDF parsing
            yield "🔄 Step 1/5: Parsing PDF document..."
            if not self.load_and_parse_pdf(pdf_path):
                yield "❌ Error: PDF parsing failed."
                return
            yield f"✅ Step 1/5: PDF Parsing Complete ({len(self.nodes)} text chunks extracted)."

            # Step 2: Contextualization
            yield "🔄 Step 2/5: Generating contextual summaries for each chunk..."
            if not await self.contextualize_chunks():
                yield "❌ Error: Contextualization failed."
                return
            yield "✅ Step 2/5: Contextualization Complete."

            # Step 3: Knowledge extraction
            yield "🔄 Step 3/5: Extracting knowledge graph (entities & relationships)..."
            if not await self.extract_knowledge_graph():
                yield "❌ Error: Knowledge graph extraction failed."
                return
            yield "✅ Step 3/5: Knowledge Graph Extraction Complete."

            # Step 4: Neo4j population
            yield "🔄 Step 4/5: Populating Neo4j Graph Database..."
            if not self.populate_neo4j():
                yield "❌ Error: Neo4j population failed."
                return
            yield "✅ Step 4/5: Neo4j Population Complete."

            # Step 5: Vector store population
            yield "🔄 Step 5/5: Populating Qdrant Vector Database..."
            if not self.populate_vector_store():
                yield "⚠️ Warning: Vector store population failed. Vector search will be unavailable."
            else:
                yield "✅ Step 5/5: Vector Store Population Complete."

            total_time = time.time() - start_time
            yield f"🎉 Pipeline completed successfully in {total_time:.2f}s! Ready for questions."

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            yield f"❌ An unexpected error occurred during the pipeline: {e}"

    async def interactive_query_loop(self, conversation_id: str = None):
        """Run interactive query loop with streaming responses and conversation memory."""
        logger.info("🔎 Starting Interactive Query Mode")

        # Generate conversation ID if not provided
        if conversation_id is None:
            conversation_id = f"session_{int(time.time())}"

        print("\n=== Lyzr RAG Query Interface ===")
        print("Ask questions about the processed document.")
        print("Type 'exit' to quit, 'bye' to end conversation, 'stats' for system statistics.\n")

        try:
            while True:
                user_input = input("Your question: ").strip()

                if user_input.lower() in ['exit', 'quit']:
                    break
                elif user_input.lower() == 'bye':
                    print("👋 Thanks for chatting! Conversation ended.")
                    break
                elif user_input.lower() == 'stats':
                    stats = self.query_router.get_routing_stats()
                    print(f"\nSystem Statistics:\n{stats}\n")
                    continue
                elif user_input.lower() == 'clear':
                    print("🧹 Conversation memory cleared.")
                    continue
                elif not user_input:
                    continue

                # Process query with streaming and conversation memory
                print("🤔 Thinking...")
                result = None

                try:
                    # Use the query agent with conversation memory
                    from agents.query_agent import QueryAgent
                    agent = QueryAgent()

                    # Get conversation context from memory
                    conversation_context = {}
                    if conversation_id:
                        conversation_context = self._get_conversation_context(conversation_id, user_input)

                    # Process query with context
                    agent_result = agent.process_query(user_input, conversation_context)

                    # Store interaction in memory
                    if conversation_id:
                        self._store_conversation_interaction(conversation_id, user_input, agent_result)

                    result = agent_result

                except Exception as e:
                    logger.error(f"Error processing query: {e}")
                    result = {
                        'query': user_input,
                        'answer': f"❌ Error processing query: {str(e)}",
                        'routing': {'method': 'error', 'confidence': 0.0},
                        'method_used': 'error'
                    }

                if result:
                    # Display final result
                    print("\n" + "="*60)
                    print(f"Query: {result.get('query', user_input)}")
                    print(f"Method: {result.get('routing', {}).get('method', 'unknown')}")
                    print(f"Confidence: {result.get('routing', {}).get('confidence', 0.0):.2f}")
                    print("-" * 60)
                    print(f"Answer: {result.get('answer', 'No answer generated')}")
                    print("="*60 + "\n")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            logger.error(f"Error in query loop: {e}")
            print(f"❌ Error: {e}")

    def interactive_query_loop_sync(self, conversation_id: str = None):
        """Run interactive query loop in sync mode with conversation memory."""
        # Use nest_asyncio to run async code in sync context
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except RuntimeError:
            pass  # Already applied

        # Run the async loop with conversation memory
        asyncio.run(self.interactive_query_loop(conversation_id))

    def _get_conversation_context(self, conversation_id: str, current_query: str) -> Dict[str, Any]:
        """Get conversation context from memory."""
        try:
            from utils.memory_manager import memory_manager

            # Get recent conversation history
            history = memory_manager.get_conversation_history(conversation_id)

            # Get conversation context
            context = memory_manager.get_conversation_context(conversation_id) or {}

            return {
                'conversation_history': history,
                'conversation_context': context,
                'current_query': current_query,
                'conversation_id': conversation_id
            }
        except Exception as e:
            logger.error(f"Error getting conversation context: {e}")
            return {'current_query': current_query, 'conversation_id': conversation_id}

    def _store_conversation_interaction(self, conversation_id: str, query: str, result: Dict[str, Any]) -> None:
        """Store conversation interaction in memory."""
        try:
            from utils.memory_manager import memory_manager

            # Store conversation context
            context_data = {
                'last_query': query,
                'last_answer': result.get('answer', ''),
                'last_method': result.get('routing', {}).get('method', 'unknown'),
                'last_confidence': result.get('routing', {}).get('confidence', 0.0)
            }
            memory_manager.store_conversation_context(conversation_id, context_data)

            # Add to conversation history
            memory_manager.add_to_conversation_history(
                conversation_id,
                {
                    'query': query,
                    'answer': result.get('answer', ''),
                    'method': result.get('routing', {}).get('method', 'unknown'),
                    'confidence': result.get('routing', {}).get('confidence', 0.0)
                }
            )

        except Exception as e:
            logger.error(f"Error storing conversation interaction: {e}")

def main():
    """Main entry point."""
    # Setup logging
    setup_logging()

    # Initialize database connections once at startup
    logger.info("Initializing database connections...")
    if not init_database_connections():
        logger.error("Failed to initialize database connections")
        return

    # Register shutdown handler to cleanup connections
    atexit.register(shutdown_database_connections)

    # Validate configuration
    try:
        settings.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--gradio":
        # Launch Gradio UI
        print("🚀 Starting Modular RAG System with Gradio UI")
        print("=" * 60)

        # Import and run Gradio interface
        from ui.gradio_app import main as gradio_main
        gradio_main()
        return

    # Log system info
    system_info = get_system_info()
    logger.info(f"System initialized: {system_info}")

    # Initialize RAG system
    rag_system = LyzrRAGSystem()

    # Check if we should run pipeline or just query
    if len(sys.argv) > 1 and sys.argv[1] == "--query-only":
        logger.info("Running in query-only mode")
        rag_system.initialize_query_systems()
        rag_system.interactive_query_loop_sync()
    else:
        # Run full pipeline - check for runtime PDF upload
        pdf_path = settings.PDF_FILE_PATH

        # Check if the default path exists, if not check data folder for uploaded PDFs
        if pdf_path == "/path/to/your/document.pdf" or not os.path.exists(pdf_path):
            # Check data folder for uploaded PDFs
            data_folder = os.path.join(os.getcwd(), "data")
            if os.path.exists(data_folder):
                pdf_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.pdf')]
                if pdf_files:
                    # Use the first PDF found in data folder
                    pdf_path = os.path.join(data_folder, pdf_files[0])
                    print(f"📄 Found uploaded PDF: {pdf_files[0]}")
                else:
                    print("❌ No PDF found in data folder")
                    print("💡 Please upload a PDF file to the data folder first")
                    print("   Or run the Gradio UI: python -m main --gradio")
                    return
            else:
                print("❌ Data folder not found and no PDF_FILE_PATH configured")
                print("💡 Please create a 'data' folder and upload your PDF there")
                print("   Or run the Gradio UI: python -m main --gradio")
                return

        print(f"📄 Processing PDF: {pdf_path}")
        success = asyncio.run(rag_system.run_full_pipeline(pdf_path))
        if success:
            # Start interactive query mode with streaming
            print("\n🚀 Launching Visual Ontology Editor (optional)...")
            print("📊 Editor available at: http://localhost:7860")
            print("🎯 Starting interactive query mode with streaming responses...")

            rag_system.interactive_query_loop_sync()
        else:
            logger.error("Pipeline failed, cannot start query mode")

if __name__ == "__main__":
    main()