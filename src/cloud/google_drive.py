"""
Google Drive integration for hybrid Codespace + Colab workflow.

This module handles:
- Uploading models to Google Drive from Colab
- Downloading models from Drive to Codespaces
- Syncing training results
- Managing model versions
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class GoogleDriveManager:
    """Manage model and data storage on Google Drive."""
    
    BASE_FOLDER = "Banking LLM"
    MODELS_FOLDER = "models"
    RESULTS_FOLDER = "results"
    DATASETS_FOLDER = "datasets"
    
    def __init__(self, drive_service=None):
        """
        Initialize Drive manager.
        
        Args:
            drive_service: Google Drive API service (from google.colab import drive)
        """
        self.drive_service = drive_service
        self.base_path = Path("/content/drive/MyDrive") if drive_service else None
    
    def get_model_path(self, model_name: str, version: str = "current") -> Path:
        """Get path to model on Google Drive."""
        if not self.base_path:
            raise RuntimeError("Drive not mounted. Use in Google Colab.")
        
        path = self.base_path / self.BASE_FOLDER / self.MODELS_FOLDER / model_name / version
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def upload_model(self, local_path: str, model_name: str, version: str = "latest") -> Dict:
        """
        Upload trained model from local to Google Drive.
        
        Args:
            local_path: Local path to model file/folder
            model_name: Name for model storage (e.g., 'banking-lora-v1')
            version: Version tag (e.g., 'latest', '1.0', 'deployed')
            
        Returns:
            Dict with upload metadata
        """
        if not self.drive_service:
            return {"status": "skipped", "reason": "Not running in Colab"}
        
        try:
            local_file = Path(local_path)
            if not local_file.exists():
                raise FileNotFoundError(f"Local file not found: {local_path}")
            
            drive_path = self.get_model_path(model_name, version)
            
            # For single files
            if local_file.is_file():
                import shutil
                dest = drive_path / local_file.name
                shutil.copy2(local_file, dest)
                logger.info(f"✓ Uploaded model: {model_name}/{version}/{local_file.name}")
                
                return {
                    "status": "success",
                    "model_name": model_name,
                    "version": version,
                    "path": str(dest),
                    "file_size_mb": local_file.stat().st_size / 1024 / 1024
                }
            
            # For directories
            else:
                import shutil
                dest = drive_path / local_file.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(local_file, dest)
                logger.info(f"✓ Uploaded model directory: {model_name}/{version}")
                
                return {
                    "status": "success",
                    "model_name": model_name,
                    "version": version,
                    "path": str(dest),
                    "type": "directory"
                }
        
        except Exception as e:
            logger.error(f"✗ Upload failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def download_model(self, model_name: str, local_path: str, version: str = "latest") -> Dict:
        """
        Download model from Google Drive to local Codespace.
        
        Args:
            model_name: Model name on Drive
            local_path: Where to save locally
            version: Which version to download
            
        Returns:
            Dict with download metadata
        """
        try:
            if not self.base_path:
                # Simulate for local development
                logger.info(f"Demo mode: Model {model_name} would be downloaded from Drive")
                return {
                    "status": "demo",
                    "message": f"In production, would download {model_name}/{version} to {local_path}"
                }
            
            drive_path = self.get_model_path(model_name, version)
            
            if not drive_path.exists():
                raise FileNotFoundError(f"Model not found on Drive: {model_name}/{version}")
            
            import shutil
            local_dir = Path(local_path)
            local_dir.mkdir(parents=True, exist_ok=True)
            
            if drive_path.is_file():
                dest = local_dir / drive_path.name
                shutil.copy2(drive_path, dest)
            else:
                dest = local_dir / drive_path.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(drive_path, dest)
            
            logger.info(f"✓ Downloaded model: {model_name}/{version} to {local_path}")
            
            return {
                "status": "success",
                "model_name": model_name,
                "version": version,
                "local_path": str(dest)
            }
        
        except Exception as e:
            logger.error(f"✗ Download failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def list_models(self) -> List[str]:
        """List all available models in Google Drive."""
        if not self.base_path:
            return []
        
        models_path = self.base_path / self.BASE_FOLDER / self.MODELS_FOLDER
        if not models_path.exists():
            return []
        
        models = [d.name for d in models_path.iterdir() if d.is_dir()]
        return sorted(models)
    
    def save_metadata(self, model_name: str, metadata: Dict, version: str = "latest"):
        """Save model metadata (training results, config, etc.)."""
        if not self.base_path:
            return
        
        metadata_path = self.get_model_path(model_name, version) / "metadata.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"✓ Saved metadata: {metadata_path}")
    
    def load_metadata(self, model_name: str, version: str = "latest") -> Optional[Dict]:
        """Load model metadata."""
        if not self.base_path:
            return None
        
        metadata_path = self.get_model_path(model_name, version) / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        
        return None


def init_drive_manager():
    """Initialize Drive manager for use in Colab."""
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        logger.info("✓ Google Drive mounted")
        return GoogleDriveManager(drive_service=drive)
    except ImportError:
        logger.info("→ Not running in Colab, using local mode")
        return GoogleDriveManager()
    except Exception as e:
        logger.error(f"✗ Failed to initialize Drive: {e}")
        return GoogleDriveManager()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # In Colab:
    # manager = init_drive_manager()
    # manager.upload_model("./trained_model", "banking-lora", "v1")
    
    # In Codespaces:
    # manager = GoogleDriveManager()
    # result = manager.download_model("banking-lora", "./models", "v1")
