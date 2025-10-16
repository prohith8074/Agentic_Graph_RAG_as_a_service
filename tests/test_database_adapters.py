"""
# Tests for database adapters (Neo4j and Neptune).
# NOTE: Neptune tests are commented out since Neptune support is disabled.
"""

import pytest
from unittest.mock import Mock, patch
from database_adapters.neo4j_adapter import Neo4jAdapter
# from database_adapters.neptune_adapter import NeptuneAdapter  # Commented out - Neptune support disabled
from database_adapters.database_factory import DatabaseFactory, DatabaseManager

class TestNeo4jAdapter:
    """Test Neo4j adapter functionality."""

    @patch('database_adapters.neo4j_adapter.GraphDatabase')
    def test_connect_success(self, mock_graph_db):
        """Test successful Neo4j connection."""
        # Mock the driver and verify_connectivity
        mock_driver = Mock()
        mock_graph_db.driver.return_value = mock_driver

        adapter = Neo4jAdapter()
        result = adapter.connect()

        assert result is True
        assert adapter.driver == mock_driver
        mock_graph_db.driver.assert_called_once()
        mock_driver.verify_connectivity.assert_called_once()

    @patch('database_adapters.neo4j_adapter.GraphDatabase')
    def test_connect_failure(self, mock_graph_db):
        """Test Neo4j connection failure."""
        mock_graph_db.driver.side_effect = Exception("Connection failed")

        adapter = Neo4jAdapter()
        result = adapter.connect()

        assert result is False
        assert adapter.driver is None

    def test_execute_query(self):
        """Test Cypher query execution."""
        adapter = Neo4jAdapter()
        adapter.driver = Mock()

        # Mock session and result
        mock_session = Mock()
        mock_result = Mock()
        mock_record = Mock()
        mock_record.data.return_value = {"name": "Test", "type": "Entity"}

        adapter.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        adapter.driver.session.return_value.__exit__ = Mock(return_value=None)
        mock_session.run.return_value = mock_result
        mock_result.data.return_value = [mock_record.data()]

        result = adapter.execute_query("MATCH (n) RETURN n")

        assert len(result) == 1
        assert result[0] == {"name": "Test", "type": "Entity"}
        mock_session.run.assert_called_once_with("MATCH (n) RETURN n")

    def test_populate_graph(self):
        """Test graph population."""
        adapter = Neo4jAdapter()
        adapter.driver = Mock()

        mock_session = Mock()
        adapter.driver.session.return_value.__enter__ = Mock(return_value=mock_session)
        adapter.driver.session.return_value.__exit__ = Mock(return_value=None)

        graph_data = {
            "entities": [{"name": "Test", "type": "Entity"}],
            "relationships": []
        }

        result = adapter.populate_graph(graph_data)

        assert result is True
        # Verify MERGE query was called
        assert mock_session.run.called

# class TestNeptuneAdapter:
#     """Test Neptune adapter functionality."""

#     @patch('database_adapters.neptune_adapter.client')
#     def test_connect_success(self, mock_client):
#         """Test successful Neptune connection."""
#         mock_client_instance = Mock()
#         mock_client_instance.submit.return_value.all.return_value.result.return_value = [1]
#         mock_client.return_value = mock_client_instance

#         adapter = NeptuneAdapter(endpoint="test-endpoint")
#         result = adapter.connect()

#         assert result is True
#         assert adapter.client == mock_client_instance

#     def test_execute_query_gremlin(self):
#         """Test Gremlin query execution."""
#         adapter = NeptuneAdapter(endpoint="test-endpoint")
#         adapter.client = Mock()
#         adapter.client.submit.return_value.all.return_value.result.return_value = [
#             {"id": "123", "label": "Test"}
#         ]

#         result = adapter.execute_query("g.V().limit(1)")

#         assert len(result) == 1
#         assert result[0]["value"] == {"id": "123", "label": "Test"}

#     def test_wrong_query_language(self):
#         """Test error when using wrong query language."""
#         adapter = NeptuneAdapter(endpoint="test-endpoint")

#         with pytest.raises(ValueError, match="only supports"):
#             adapter.execute_query("MATCH (n) RETURN n", "cypher")

class TestDatabaseFactory:
    """Test database factory functionality."""

    def test_create_neo4j_adapter(self):
        """Test Neo4j adapter creation."""
        with patch('database_adapters.database_factory.Neo4jAdapter') as mock_adapter:
            mock_instance = Mock()
            mock_adapter.return_value = mock_instance

            adapter = DatabaseFactory.create_adapter("neo4j")

            assert adapter == mock_instance
            mock_adapter.assert_called_once()

    # def test_create_neptune_adapter(self):
    #     """Test Neptune adapter creation."""
    #     with patch('database_adapters.database_factory.NeptuneAdapter') as mock_adapter:
    #         mock_instance = Mock()
    #         mock_adapter.return_value = mock_instance

    #         adapter = DatabaseFactory.create_adapter("neptune", endpoint="test-endpoint")

    #         assert adapter == mock_instance
    #         mock_adapter.assert_called_once_with(endpoint="test-endpoint", port=8182, use_ssl=True)

    def test_unsupported_adapter(self):
        """Test unsupported database type."""
        adapter = DatabaseFactory.create_adapter("unsupported")

        assert adapter is None

class TestDatabaseManager:
    """Test database manager functionality."""

    def test_register_and_set_active(self):
        """Test adapter registration and activation."""
        manager = DatabaseManager()
        mock_adapter = Mock()

        result = manager.register_adapter("test", mock_adapter)
        assert result is True

        result = manager.set_active_adapter("test")
        assert result is True
        assert manager.active_adapter == "test"

    def test_get_adapter(self):
        """Test adapter retrieval."""
        manager = DatabaseManager()
        mock_adapter = Mock()

        manager.register_adapter("test", mock_adapter)
        manager.set_active_adapter("test")

        adapter = manager.get_adapter()
        assert adapter == mock_adapter

        adapter = manager.get_adapter("test")
        assert adapter == mock_adapter

    def test_get_nonexistent_adapter(self):
        """Test retrieval of non-existent adapter."""
        manager = DatabaseManager()

        adapter = manager.get_adapter("nonexistent")
        assert adapter is None