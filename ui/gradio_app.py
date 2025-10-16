"""
Gradio UI for the Modular RAG System.
Provides PDF upload, processing, and interactive querying interface.
"""

import gradio as gr
import os
import asyncio
import logging
from typing import Optional
import shutil
import networkx as nx
import matplotlib.pyplot as plt
from config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGInterface:
    """Interface for the RAG system with dynamic path assignment."""

    def __init__(self):
        """Initialize RAG interface."""
        self.rag_system = None
        self.pdf_path = None
        self.processing_status = "Ready"

    def load_environment(self):
        """Load environment variables from .env file."""
        try:
            from dotenv import load_dotenv
            load_dotenv('.env')
            logger.info("Environment variables loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to load environment: {e}")
            return False

    def setup_pdf_path(self, uploaded_file) -> str:
        """
        Set up PDF path dynamically from uploaded file.

        Args:
            uploaded_file: Gradio file object

        Returns:
            Path to the uploaded PDF
        """
        try:
            # Create data folder if it doesn't exist
            data_folder = os.path.join(os.getcwd(), "data")
            os.makedirs(data_folder, exist_ok=True)

            # Get the original filename from the uploaded file
            # uploaded_file can be a temp path string or a file-like object
            if hasattr(uploaded_file, 'name'):
                # If it's a file-like object, get the name
                original_name = os.path.basename(uploaded_file.name)
            else:
                # If it's a string path, extract filename
                original_name = os.path.basename(str(uploaded_file))

            # Save uploaded file with unique name to avoid conflicts
            import time
            unique_name = f"{int(time.time())}_{original_name}"
            file_path = os.path.join(data_folder, unique_name)

            # Copy the uploaded file
            shutil.copy(str(uploaded_file), file_path)

            logger.info(f"PDF saved to: {file_path}")

            # Update environment variables dynamically
            os.environ['PDF_FILE_PATH'] = file_path

            # Also set contextual data path if needed
            contextual_path = os.path.join(data_folder, "contextual_data.txt")
            os.environ['CONTEXTUAL_DATA_PATH'] = contextual_path

            return file_path
  
        except Exception as e:
            logger.error(f"Error setting up PDF path: {e}")
            raise

    async def process_pdf(self, uploaded_file):
        """
        Process uploaded PDF, yielding status updates.
        """
        try:
            if not uploaded_file:
                yield "❌ Please upload a PDF file first"
                return

            yield "1. Setting up file path..."
            self.pdf_path = self.setup_pdf_path(uploaded_file)

            yield "2. Loading environment and initializing RAG system..."
            if not self.load_environment():
                yield "❌ Failed to load environment variables"
                return

            from main import LyzrRAGSystem
            self.rag_system = LyzrRAGSystem()

            yield "3. Starting RAG pipeline..."
            
            # Run pipeline and yield status updates
            async for status in self.rag_system.run_full_pipeline(self.pdf_path):
                yield status

        except Exception as e:
            error_msg = f"❌ An unexpected error occurred: {str(e)}"
            logger.error(error_msg, exc_info=True)
            yield error_msg

    def answer_question(self, question: str) -> str:
        """
        Answer user question about the processed PDF.

        Args:
            question: User question

        Returns:
            AI-generated answer
        """
        try:
            if not self.rag_system:
                return "❌ Please process a PDF first"

            if not question.strip():
                return "❌ Please ask a question"

            # Use the query router for intelligent query processing (synchronous version)
            from query.query_router import QueryRouter

            router = QueryRouter()
            result = router.route_query(question)

            answer = result.get('answer', '❌ No answer generated')
            return answer

        except Exception as e:
            return f"❌ Error answering question: {str(e)}"

    def get_system_stats(self) -> str:
        """Get system statistics."""
        try:
            from query.query_router import QueryRouter
            router = QueryRouter()
            stats = router.get_routing_stats()

            # Add path information
            current_pdf = os.environ.get('PDF_FILE_PATH', 'Not set')
            current_contextual = os.environ.get('CONTEXTUAL_DATA_PATH', 'Not set')

            return f"""
📊 **System Statistics:**

🔗 **Database Connections:**
- Neo4j Connected: {stats.get('neo4j_connected', False)}
- Vector Store Available: {stats.get('vector_stats', {}).get('total_vectors', 0) > 0}

📁 **Current Paths:**
- PDF Path: {current_pdf}
- Contextual Data: {current_contextual}

📈 **Performance:**
- Router Config: {stats.get('router_config', {})}

🔧 **Available Plugins:**
- Database: {', '.join(['Neo4j', 'Neptune'])}
- Query: {', '.join(['Vector', 'Graph', 'Hybrid', 'Filter'])}
            """
        except Exception as e:
            return f"❌ Error getting stats: {str(e)}"

# Global RAG interface instance
rag_interface = RAGInterface()

def create_gradio_interface():
    """Create Gradio interface for the RAG system."""

    with gr.Blocks(title="Modular RAG System", theme=gr.themes.Soft()) as interface:

        gr.Markdown("""
        # 🚀 Modular RAG System

        **Intelligent Document Processing & Querying**

        Upload a PDF document and ask questions about its content using our modular RAG system with:
        - 🔍 Intelligent Query Routing (Vector/Graph/Hybrid/Filter)
        - 🧠 LLM Orchestration (Groq, Cohere, Nomic)
        - 📊 Plugin Architecture for Easy Extension
        - 🔗 Knowledge Graph Generation
        - 💾 Dual Memory Systems (Redis + Qdrant)
        """)

        with gr.Row():
            # Left column - Upload and Processing
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Document Upload")

                pdf_upload = gr.File(
                    label="Upload PDF Document",
                    file_types=[".pdf"]
                )

                process_btn = gr.Button(
                    "🔄 Process PDF",
                    variant="primary",
                    size="lg"
                )

                processing_output = gr.Textbox(
                    label="Processing Status",
                    lines=3,
                    interactive=False
                )

            # Right column - Query Interface
            with gr.Column(scale=1):
                gr.Markdown("### ❓ Ask Questions")

                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask anything about the processed document...",
                    lines=2
                )

                ask_btn = gr.Button(
                    "🤔 Ask Question",
                    variant="secondary"
                )

                # Streaming response box
                streaming_status = gr.Textbox(
                    label="🔄 Live Processing Status",
                    lines=8,
                    interactive=False,
                    value="Ready to process queries..."
                )

                # Wrapper function for Gradio with streaming support
                async def async_answer_question(question: str) -> tuple[str, str]:
                    """Asynchronous wrapper for answer_question with streaming and conversation memory."""
                    status_updates = []
                    try:
                        if not rag_interface.rag_system:
                            return "❌ Please process a PDF first", "❌ Please process a PDF first"

                        if not question.strip():
                            return "❌ Please ask a question", "❌ Please ask a question"

                        # Generate conversation ID
                        conversation_id = f"session_{hash(question) % 10000}"
                        status_updates.append("🔍 Starting query processing...")

                        # Process query with streaming
                        result = None
                        step_count = 0

                        async for query_result in rag_interface.rag_system.advanced_router.route_query_advanced(question):
                            step_count += 1

                            if query_result.metadata.get('intermediate'):
                                # Intermediate step
                                status = query_result.metadata.get('status', 'Processing...')
                                status_updates.append(f"📊 Step {step_count}: {status}")
                            else:
                                # Final result
                                result = query_result
                                method_used = result.method_used.upper()
                                confidence = result.confidence_score
                                status_updates.append(f"✅ Final: Using {method_used} search (confidence: {confidence:.2f})")
                                break

                            # Yield intermediate status
                            current_status = "\n".join(status_updates[-5:])  # Show last 5 updates
                            # Note: Gradio streaming would need proper implementation

                        if result:
                            # Store interaction in memory
                            from utils.memory_manager import memory_manager
                            memory_manager.add_to_conversation_history(
                                conversation_id,
                                {
                                    'query': question,
                                    'answer': result.final_answer,
                                    'method': result.method_used,
                                    'confidence': result.confidence_score
                                }
                            )

                            final_status = "\n".join(status_updates[-5:]) + f"\n🎯 Method: {result.method_used.upper()}"
                            return result.final_answer, final_status
                        else:
                            return '❌ No answer generated', '❌ Query processing failed'

                    except Exception as e:
                        error_status = "\n".join(status_updates) + f"\n❌ Error: {str(e)}"
                        return f"❌ Error answering question: {str(e)}", error_status

                answer_output = gr.Textbox(
                    label="Answer",
                    lines=5,
                    interactive=False
                )

                # Ontology Editor Section
                with gr.Accordion("🧠 Ontology Editor & Graph Visualizer", open=False):
                    gr.Markdown("**View, analyze, and edit the knowledge graph ontology with interactive graph visualization and LLM-powered discussions.**")

                    ontology_display = gr.JSON(
                        label="📊 Current Ontology (Entities & Relationships)",
                        value=None
                    )

                    with gr.Row():
                        load_ontology_btn = gr.Button("📚 Load Ontology from Processed PDF", variant="secondary")
                        suggest_improvements_btn = gr.Button("🤖 AI-Powered Suggestions (Cohere)", variant="secondary")
                        visualize_graph_btn = gr.Button("📈 Visualize Graph", variant="secondary")
                        download_graph_btn = gr.Button("📥 Download Graph", variant="secondary")

                    # Graph Visualization
                    graph_visualization = gr.Plot(
                        label="🕸️ Knowledge Graph Visualization",
                    )

                    improvement_suggestions = gr.JSON(
                        label="💡 AI Improvement Suggestions",
                        value=None
                    )

                    # Interactive Discussion Section
                    with gr.Accordion("💬 Discuss Graph Changes with LLM", open=False):
                        gr.Markdown("**Have a conversation with the LLM about graph connections and proposed changes.**")

                        discussion_input = gr.Textbox(
                            label="💭 Your Question/Change Request",
                            placeholder="e.g., 'Why is Python connected to Data Science?', 'Add a connection between ML and AI', 'Remove this relationship'...",
                            lines=2
                        )

                        with gr.Row():
                            discuss_btn = gr.Button("🤔 Discuss with LLM", variant="secondary")
                            analyze_connections_btn = gr.Button("🔍 Analyze Connections", variant="secondary")

                        discussion_response = gr.Textbox(
                            label="🧠 LLM Analysis & Suggestions",
                            lines=6,
                            interactive=False
                        )

                    with gr.Row():
                        apply_suggestion_btn = gr.Button("✅ Apply Selected Suggestion", variant="primary")
                        selected_suggestion = gr.Dropdown(
                            label="Select Suggestion to Apply",
                            choices=[],
                            value=None
                        )

                    edit_result = gr.Textbox(
                        label="📝 Edit Result & Status",
                        lines=3,
                        interactive=False
                    )

                    with gr.Accordion("🕰️ Ontology Versioning", open=False):
                        gr.Markdown("**Track and revert to previous versions of your ontology.**")

                        with gr.Row():
                            ontology_version_dropdown = gr.Dropdown(
                                label="Select Ontology Version",
                                choices=[],
                                value=None
                            )
                            switch_version_btn = gr.Button("🔄 Switch to Version", variant="secondary")

                        version_info_display = gr.JSON(
                            label="📜 Version Information",
                            value=None
                        )

                    gr.Markdown("""
                    **How to use the Ontology Editor:**

                    📚 **Basic Usage:**
                    1. Process your PDF first
                    2. Click "Load Ontology" to view the generated knowledge graph
                    3. Use "AI-Powered Suggestions" to get intelligent improvement recommendations
                    4. Select and apply suggestions to refine your ontology

                    📈 **Graph Visualization:**
                    - Click "Visualize Graph" to see interactive node-link diagram
                    - Hover over nodes to see details, drag to rearrange
                    - Understand how entities are connected in your knowledge graph

                    💬 **LLM Discussion:**
                    - Ask questions about connections: "Why are these connected?"
                    - Propose changes: "Add relationship between X and Y"
                    - Get analysis: "What would happen if I remove this connection?"
                    - Discuss improvements: "How can I better organize this graph?"
                    """)

        file_output = gr.File(label="Download Graph")

        # Ontology editor and graph visualization
        ontology_editor = None
        current_graph_data = None
        current_suggestions_state = gr.State()

        # Bottom section - System Info
        with gr.Row():
            stats_btn = gr.Button("📊 System Statistics")
            stats_output = gr.Textbox(
                label="System Information",
                lines=8,
                interactive=False
            )

        # Ontology editor and graph visualization
        ontology_editor = None
        current_graph_data = None

        def load_ontology_data():
            """Load ontology from processed knowledge graph."""
            try:
                if not rag_interface.rag_system or not rag_interface.rag_system.knowledge_graph:
                    return None, "❌ Please process a PDF first to generate the ontology"

                nonlocal ontology_editor
                from knowledge_graph.ontology_editor import OntologyEditor
                ontology_editor = OntologyEditor()

                # Load the knowledge graph as ontology
                success = ontology_editor.load_ontology(rag_interface.rag_system.knowledge_graph)
                if success:
                    return ontology_editor.current_ontology, "✅ Ontology loaded successfully"
                else:
                    return None, "❌ Failed to load ontology"

            except Exception as e:
                return None, f"❌ Error loading ontology: {str(e)}"

        # Global variable to store suggestions
        current_suggestions = None

        def suggest_ontology_improvements():
            """Generate ontology improvement suggestions.

            ### Tools Used
            *   `ontology_editor.suggest_improvements`
            """
            try:
                if not ontology_editor:
                    return None, gr.update(choices=[], value=None), "❌ Please load ontology first", None

                import asyncio
                suggestions = asyncio.run(ontology_editor.suggest_improvements())

                # Create dropdown choices from suggestions
                choices = []
                if suggestions and 'suggestions' in suggestions:
                    for i, suggestion in enumerate(suggestions['suggestions']):
                        desc = suggestion.get('description', f'Suggestion {i+1}')
                        choices.append(f"{i+1}: {desc}")

                return suggestions, gr.update(choices=choices, value=None), "✅ Suggestions generated", suggestions

            except Exception as e:
                return None, gr.update(choices=[], value=None), f"❌ Error generating suggestions: {str(e)}", None

        def apply_selected_suggestion(suggestion_idx, current_suggestions):
            """Apply a selected suggestion.

            ### Tools Used
            *   `ontology_editor.apply_suggestion`
            """
            if not ontology_editor or not current_suggestions:
                return "❌ Please generate suggestions first", gr.update()

            if not suggestion_idx:
                return "❌ Please select a suggestion", gr.update()

            try:
                # Parse suggestion index
                idx = int(suggestion_idx.split(':')[0]) - 1

                if 'suggestions' not in current_suggestions or idx >= len(current_suggestions['suggestions']):
                    return "❌ Invalid suggestion index", gr.update()

                suggestion = current_suggestions['suggestions'][idx]

                # Apply the suggestion
                import asyncio
                success, message, edit = asyncio.run(ontology_editor.apply_suggestion(suggestion))

                if success:
                    print(f"Version updated to: {ontology_editor.current_version_id}")
                    version_ids = [v.version_id for v in ontology_editor.versions]
                    return f"✅ {message}", gr.update(choices=version_ids, value=ontology_editor.current_version_id)
                else:
                    return f"❌ {message}", gr.update()

            except Exception as e:
                return f"❌ Error applying suggestion: {str(e)}", gr.update()

        def get_ontology_versions():
            """Get list of available ontology versions."""
            if not ontology_editor:
                return gr.update(choices=[]), None

            version_ids = [v.version_id for v in ontology_editor.versions]
            return gr.update(choices=version_ids, value=ontology_editor.current_version_id), None

        def switch_ontology_version(version_id):
            """Switch to a selected ontology version.

            ### Tools Used
            *   `ontology_editor.switch_to_version`
            *   `load_ontology_data`
            *   `visualize_ontology_graph`
            *   `ontology_editor.get_version_info`
            """
            if not ontology_editor:
                return None, None, None, "❌ Ontology not loaded"

            success, message = ontology_editor.switch_to_version(version_id)
            if success:
                # Reload ontology data and visualization
                ontology_data, _ = load_ontology_data()
                graph_visualization_data = visualize_ontology_graph()
                version_info = ontology_editor.get_version_info(version_id)
                return ontology_data, graph_visualization_data, version_info, f"✅ Switched to version {version_id}"
            else:
                return None, None, None, f"❌ {message}"

        def download_graph():
            """Export the current ontology to a JSON file and return the path.

            ### Tools Used
            *   `ontology_editor.export_ontology`
            """
            if not ontology_editor:
                return None

            try:
                # Export the ontology to a JSON string
                ontology_json = ontology_editor.export_ontology(format="json")

                # Save the JSON to a temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", prefix="ontology_") as f:
                    f.write(ontology_json)
                    return gr.File(f.name, visible=True)
            except Exception as e:
                logger.error(f"Error exporting graph: {e}")
                return None
    
            # Event handlers
        process_btn.click(
            fn=rag_interface.process_pdf,
            inputs=[pdf_upload],
            outputs=[processing_output]
        )

        ask_btn.click(
            fn=async_answer_question,
            inputs=[question_input],
            outputs=[answer_output, streaming_status]
        )

        stats_btn.click(
            fn=rag_interface.get_system_stats,
            inputs=[],
            outputs=[stats_output]
        )

        # Ontology editor event handlers
        load_ontology_btn.click(
            fn=load_ontology_data,
            inputs=[],
            outputs=[ontology_display, edit_result]
        ).then(
            fn=get_ontology_versions,
            inputs=[],
            outputs=[ontology_version_dropdown, version_info_display]
        )

        suggest_improvements_btn.click(
            fn=suggest_ontology_improvements,
            inputs=[],
            outputs=[improvement_suggestions, selected_suggestion, edit_result, current_suggestions_state]
        )

        apply_suggestion_btn.click(
            fn=apply_selected_suggestion,
            inputs=[selected_suggestion, current_suggestions_state],
            outputs=[edit_result, ontology_version_dropdown]
        )

        switch_version_btn.click(
            fn=switch_ontology_version,
            inputs=[ontology_version_dropdown],
            outputs=[ontology_display, graph_visualization, version_info_display, edit_result]
        )

        download_graph_btn.click(
            fn=download_graph,
            inputs=[],
            outputs=[file_output]
        )

        # Define functions outside the button scope for proper access
        def visualize_ontology_graph():
            """Generate interactive graph visualization using Matplotlib."""
            try:
                if not ontology_editor or not ontology_editor.current_ontology:
                    # Return an empty plot with a message
                    fig, ax = plt.subplots()
                    ax.text(0.5, 0.5, "Please load an ontology first.", ha='center', va='center')
                    return fig

                entities = ontology_editor.current_ontology.get("entities", [])
                relationships = ontology_editor.current_ontology.get("relationships", [])
                
                if not entities:
                    fig, ax = plt.subplots()
                    ax.text(0.5, 0.5, "Ontology contains no entities to display.", ha='center', va='center')
                    return fig

                G = nx.DiGraph()
                for entity in entities:
                    G.add_node(entity.get("name", ""), node_type=entity.get("type", "Unknown"))

                for rel in relationships:
                    if G.has_node(rel.get("source")) and G.has_node(rel.get("target")):
                        G.add_edge(rel.get("source"), rel.get("target"), label=rel.get("label", ""))

                fig, ax = plt.subplots(figsize=(16, 10))
                pos = nx.spring_layout(G, k=0.8, iterations=50)
                
                nx.draw(G, pos, ax=ax, with_labels=True, node_size=3000, node_color="lightblue", font_size=8, arrows=True)
                edge_labels = nx.get_edge_attributes(G, 'label')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7)
                
                return fig

            except Exception as e:
                logger.error(f"Error visualizing graph: {e}", exc_info=True)
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
                return fig

        def discuss_with_llm(user_input):
            """Have an interactive discussion about the ontology with LLM."""
            try:
                if not ontology_editor:
                    return "❌ Please load ontology first"

                if not user_input.strip():
                    return "❌ Please enter a question or change request"

                # Create comprehensive context
                ontology_stats = ontology_editor.get_ontology_stats()
                ontology_summary = ontology_editor._summarize_ontology()

                discussion_prompt = f"""
                You are an expert knowledge graph analyst. The user is asking about or wants to modify their ontology.

                Current Ontology Summary:
                {ontology_summary}

                Ontology Statistics:
                - {ontology_stats['total_entities']} entities
                - {ontology_stats['total_relationships']} relationships
                - Connectivity ratio: {ontology_stats['connectivity_ratio']:.2%}

                User Request: "{user_input}"

                Provide a thoughtful analysis that:
                1. Understands their question/request
                2. Provides specific insights about current connections
                3. Suggests concrete improvements or answers their question
                4. Explains the reasoning behind your suggestions
                5. Offers actionable next steps

                Be conversational and helpful, like a knowledgeable colleague discussing the knowledge graph.
                """

                import groq
                client = groq.Groq(api_key=settings.GROQ_API_KEY)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": discussion_prompt}],
                    temperature=0.7,
                    max_tokens=1000
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                return f"❌ Error in LLM discussion: {str(e)}"

        def analyze_connections():
            """Provide automated analysis of graph connections."""
            try:
                if not ontology_editor:
                    return "❌ Please load ontology first"

                ontology_stats = ontology_editor.get_ontology_stats()

                analysis_prompt = f"""
                Analyze this knowledge graph and provide insights about its structure and connections:

                Graph Statistics:
                - {ontology_stats['total_entities']} entities
                - {ontology_stats['total_relationships']} relationships
                - {ontology_stats['connected_entities']} entities have connections
                - {ontology_stats['isolated_entities']} entities are isolated
                - Connectivity ratio: {ontology_stats['connectivity_ratio']:.2%}

                Entity Types: {ontology_stats['entity_types']}
                Relationship Types: {ontology_stats['relationship_types']}

                Provide a comprehensive analysis covering:
                1. Overall graph health and connectivity
                2. Key insights about relationships between entities
                3. Potential areas for improvement
                4. Interesting patterns or clusters
                5. Recommendations for better organization

                Be specific and reference actual entity types and relationships where possible.
                """

                import groq
                client = groq.Groq(api_key=settings.GROQ_API_KEY)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0.6,
                    max_tokens=800
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                return f"❌ Error analyzing connections: {str(e)}"

        def generate_graph_html(viz_data):
            """Generate HTML for interactive graph visualization."""
            nodes = viz_data.get('nodes', [])
            edges = viz_data.get('edges', [])

            # Create D3.js visualization
            html_template = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://d3js.org/d3.v7.min.js"></script>
                <style>
                    .node {{
                        stroke: #fff;
                        stroke-width: 2px;
                    }}
                    .link {{
                        stroke: #999;
                        stroke-opacity: 0.6;
                    }}
                    .node-text {{
                        font-family: Arial, sans-serif;
                        font-size: 12px;
                        text-anchor: middle;
                        pointer-events: none;
                    }}
                    .tooltip {{
                        position: absolute;
                        background: rgba(0, 0, 0, 0.8);
                        color: white;
                        padding: 8px;
                        border-radius: 4px;
                        font-size: 12px;
                        pointer-events: none;
                        opacity: 0;
                    }}
                </style>
            </head>
            <body>
                <div id="graph-container" style="width: 100%; height: 500px; border: 1px solid #ddd;"></div>
                <div class="tooltip" id="tooltip"></div>

                <script>
                    const width = 800;
                    const height = 500;

                    const nodes = {nodes};
                    const links = {edges};

                    const svg = d3.select("#graph-container")
                        .append("svg")
                        .attr("width", width)
                        .attr("height", height);

                    const simulation = d3.forceSimulation(nodes)
                        .force("link", d3.forceLink(links).id(d => d.id).distance(100))
                        .force("charge", d3.forceManyBody().strength(-300))
                        .force("center", d3.forceCenter(width / 2, height / 2));

                    // Create color scale for entity types
                    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

                    // Create links
                    const link = svg.append("g")
                        .selectAll("line")
                        .data(links)
                        .enter().append("line")
                        .attr("class", "link")
                        .attr("stroke-width", 2);

                    // Create nodes
                    const node = svg.append("g")
                        .selectAll("circle")
                        .data(nodes)
                        .enter().append("circle")
                        .attr("class", "node")
                        .attr("r", 20)
                        .attr("fill", d => colorScale(d.type))
                        .call(d3.drag()
                            .on("start", dragstarted)
                            .on("drag", dragged)
                            .on("end", dragended));

                    // Add labels
                    const labels = svg.append("g")
                        .selectAll("text")
                        .data(nodes)
                        .enter().append("text")
                        .attr("class", "node-text")
                        .attr("dy", 4)
                        .text(d => d.label.length > 15 ? d.label.substring(0, 12) + "..." : d.label);

                    // Add tooltips
                    const tooltip = d3.select("#tooltip");

                    node.on("mouseover", function(event, d) {{
                        tooltip.style("opacity", 1)
                               .html(`<strong>${{d.label}}</strong><br>Type: ${{d.type}}<br>Connections: ${{d.properties?.connections || 'N/A'}}`)
                               .style("left", (event.pageX + 10) + "px")
                               .style("top", (event.pageY - 10) + "px");
                    }})
                    .on("mouseout", function() {{
                        tooltip.style("opacity", 0);
                    }});

                    // Update positions
                    simulation.on("tick", () => {{
                        link.attr("x1", d => d.source.x)
                            .attr("y1", d => d.source.y)
                            .attr("x2", d => d.target.x)
                            .attr("y2", d => d.target.y);

                        node.attr("cx", d => d.x)
                            .attr("cy", d => d.y);

                        labels.attr("x", d => d.x)
                              .attr("y", d => d.y);
                    }});

                    function dragstarted(event, d) {{
                        if (!event.active) simulation.alphaTarget(0.3).restart();
                        d.fx = d.x;
                        d.fy = d.y;
                    }}

                    function dragged(event, d) {{
                        d.fx = event.x;
                        d.fy = event.y;
                    }}

                    function dragended(event, d) {{
                        if (!event.active) simulation.alphaTarget(0);
                        d.fx = null;
                        d.fy = null;
                    }}

                    // Add legend
                    const legend = svg.append("g")
                        .attr("transform", "translate(20, 20)");

                    const types = [...new Set(nodes.map(d => d.type))];
                    types.forEach((type, i) => {{
                        legend.append("circle")
                            .attr("cx", 0)
                            .attr("cy", i * 20)
                            .attr("r", 6)
                            .attr("fill", colorScale(type));

                        legend.append("text")
                            .attr("x", 15)
                            .attr("y", i * 20 + 4)
                            .text(type)
                            .style("font-size", "12px");
                    }});
                </script>
            </body>
            </html>
            """

            return html_template

        visualize_graph_btn.click(
            fn=visualize_ontology_graph,
            inputs=[],
            outputs=[graph_visualization]
        )

        discuss_btn.click(
            fn=discuss_with_llm,
            inputs=[discussion_input],
            outputs=[discussion_response]
        )

        analyze_connections_btn.click(
            fn=analyze_connections,
            inputs=[],
            outputs=[discussion_response]
        )

        apply_suggestion_btn.click(
            fn=apply_selected_suggestion,
            inputs=[selected_suggestion],
            outputs=[edit_result]
        )

        # Example questions
        gr.Markdown("""
        ### 💡 Example Questions:
        - "What are the main topics discussed?"
        - "Explain the key concepts in detail"
        - "What are the relationships between components?"
        - "Compare different approaches mentioned"
        - "Show me only the technical specifications"
        """)

    return interface

def main():
    """Main entry point for Gradio app."""
    print("🚀 Starting Modular RAG System with Gradio UI")
    print("=" * 60)

    # Load environment variables
    if not rag_interface.load_environment():
        print("❌ Failed to load environment variables")
        return

    # Create and launch interface
    interface = create_gradio_interface()

    print("🌐 Launching Gradio interface...")
    print("📱 Access the UI at: http://localhost:7860")

    # Get port from environment variable or use default
    server_port = int(os.environ.get('GRADIO_SERVER_PORT', 8000))

    # Try to find an available port if the default is in use
    import socket
    def find_available_port(start_port):
        for port in range(start_port, start_port + 200):  # Try 200 ports
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.5)  # Shorter timeout for faster checking
                    result = s.connect_ex(('127.0.0.1', port))
                    if result != 0:  # Port is available if connect_ex returns non-zero
                        # Double-check by trying to bind
                        try:
                            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                            test_socket.bind(('', port))
                            test_socket.close()
                            return port
                        except OSError:
                            continue
            except OSError:
                continue
        raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + 199}")

    try:
        available_port = find_available_port(server_port)
        if available_port != server_port:
            print(f"⚠️ Port {server_port} is in use, using port {available_port} instead")
    except RuntimeError as e:
        print(f"❌ {e}")
        print("💡 Try setting GRADIO_SERVER_PORT environment variable to a specific port")
        return

    # Launch the interface
    interface.launch(
        server_name="127.0.0.1",
        server_port=available_port,
        share=True,  # Disable public access to avoid port conflicts
        debug=False  # Disable debug mode
    )

if __name__ == "__main__":
    main()