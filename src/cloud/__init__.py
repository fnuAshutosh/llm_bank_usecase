"""Cloud integration modules for hybrid Codespace + Colab architecture."""

from .google_drive import GoogleDriveManager, init_drive_manager

__all__ = ["GoogleDriveManager", "init_drive_manager"]
