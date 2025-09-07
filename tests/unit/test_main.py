"""Tests for main.py config path functionality."""

import os
import unittest
from unittest.mock import patch, MagicMock


class TestMainConfigPath(unittest.TestCase):
    """Test config path resolution in main.py"""

    def test_config_path_default(self):
        """Test default config path when SHS_CONFIG_PATH is not set."""
        # Ensure SHS_CONFIG_PATH is not set
        if 'SHS_CONFIG_PATH' in os.environ:
            del os.environ['SHS_CONFIG_PATH']
        
        config_path = os.getenv("SHS_CONFIG_PATH", "config.yaml")
        
        self.assertEqual(config_path, "config.yaml")

    def test_config_path_environment_variable(self):
        """Test config path when SHS_CONFIG_PATH environment variable is set."""
        test_path = "/custom/path/to/my-config.yaml"
        
        # Set environment variable
        os.environ['SHS_CONFIG_PATH'] = test_path
        
        try:
            config_path = os.getenv("SHS_CONFIG_PATH", "config.yaml")
            self.assertEqual(config_path, test_path)
        finally:
            # Clean up
            del os.environ['SHS_CONFIG_PATH']

    @patch('spark_history_mcp.core.main.Config.from_file')
    @patch('spark_history_mcp.core.main.app.run')
    def test_main_uses_environment_variable(self, mock_app_run, mock_config_from_file):
        """Test that main() function uses SHS_CONFIG_PATH environment variable."""
        from spark_history_mcp.core.main import main
        
        test_config_path = "/test/path/config.yaml"
        os.environ['SHS_CONFIG_PATH'] = test_config_path
        
        # Mock config object
        mock_config = MagicMock()
        mock_config.mcp.debug = False
        mock_config_from_file.return_value = mock_config
        
        try:
            main()
            
            # Verify Config.from_file was called with the environment variable path
            mock_config_from_file.assert_called_once_with(test_config_path)
            mock_app_run.assert_called_once_with(mock_config)
            
        finally:
            # Clean up
            del os.environ['SHS_CONFIG_PATH']

    @patch('spark_history_mcp.core.main.Config.from_file')
    @patch('spark_history_mcp.core.main.app.run')
    def test_main_uses_default_config(self, mock_app_run, mock_config_from_file):
        """Test that main() function uses default config.yaml when SHS_CONFIG_PATH is not set."""
        from spark_history_mcp.core.main import main
        
        # Ensure SHS_CONFIG_PATH is not set
        if 'SHS_CONFIG_PATH' in os.environ:
            del os.environ['SHS_CONFIG_PATH']
        
        # Mock config object
        mock_config = MagicMock()
        mock_config.mcp.debug = False
        mock_config_from_file.return_value = mock_config
        
        main()
        
        # Verify Config.from_file was called with default path
        mock_config_from_file.assert_called_once_with("config.yaml")
        mock_app_run.assert_called_once_with(mock_config)

    def test_multiple_environment_variable_changes(self):
        """Test that config path changes dynamically with environment variable."""
        # Test path 1
        test_path_1 = "/path/one/config.yaml"
        os.environ['SHS_CONFIG_PATH'] = test_path_1
        config_path = os.getenv("SHS_CONFIG_PATH", "config.yaml")
        self.assertEqual(config_path, test_path_1)
        
        # Test path 2
        test_path_2 = "/path/two/different-config.yaml"
        os.environ['SHS_CONFIG_PATH'] = test_path_2
        config_path = os.getenv("SHS_CONFIG_PATH", "config.yaml")
        self.assertEqual(config_path, test_path_2)
        
        # Back to default
        del os.environ['SHS_CONFIG_PATH']
        config_path = os.getenv("SHS_CONFIG_PATH", "config.yaml")
        self.assertEqual(config_path, "config.yaml")


if __name__ == '__main__':
    unittest.main()
