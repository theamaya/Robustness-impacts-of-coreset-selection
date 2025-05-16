import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

class DatasetPaths:
    """Utility class to manage dataset paths and configurations."""
    
    def __init__(self, config_path: str = "config/dataset_paths.yaml"):
        """
        Initialize DatasetPaths with configuration file.
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load the YAML configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_path(self, dataset_name: str) -> str:
        """
        Get the path for a specific dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            str: Path to the dataset
        """
        if dataset_name not in self.config['datasets']:
            raise KeyError(f"Dataset '{dataset_name}' not found in configuration")
        return self.config['datasets'][dataset_name]['path']
    
    def get_description(self, dataset_name: str) -> str:
        """Get the description of a dataset."""
        return self.config['datasets'][dataset_name]['description']
    
    def get_source(self, dataset_name: str) -> Optional[str]:
        """Get the source repository/URL for a dataset."""
        return self.config['datasets'][dataset_name].get('source')
    
    def get_requirements(self, dataset_name: str) -> Optional[list]:
        """Get any special requirements for a dataset."""
        return self.config['datasets'][dataset_name].get('requirements')
    
    def get_structure(self, dataset_name: str) -> Optional[list]:
        """Get the directory structure of a dataset."""
        return self.config['datasets'][dataset_name].get('structure')
    
    def verify_path(self, dataset_name: str) -> bool:
        """
        Verify if the dataset path exists.
        
        Args:
            dataset_name (str): Name of the dataset
            
        Returns:
            bool: True if path exists, False otherwise
        """
        path = self.get_path(dataset_name)
        return os.path.exists(path)
    
    def list_datasets(self) -> list:
        """List all available datasets."""
        return list(self.config['datasets'].keys())

# Create a singleton instance
dataset_paths = DatasetPaths()

# Example usage:
# from deepcore.utils.dataset_paths import dataset_paths
# 
# # Get path for CMNIST
# cmnist_path = dataset_paths.get_path('cmnist')
# 
# # Get description
# description = dataset_paths.get_description('cmnist')
# 
# # List all datasets
# all_datasets = dataset_paths.list_datasets() 