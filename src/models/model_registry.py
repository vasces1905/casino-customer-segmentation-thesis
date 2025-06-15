# src/models/model_registry.py

"""
Model Registry for Academic Version Control
==========================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution

Tracks model versions for reproducible research.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Academic model registry for version control and experiment tracking.
    
    Ensures reproducibility for thesis requirements.
    """
    
    def __init__(self, registry_path: str = "model_registry.json"):
        self.registry_path = registry_path
        self.registry = self._load_registry()
        
    def _load_registry(self) -> Dict:
        """Load existing registry or create new"""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        else:
            return {
                "models": [],
                "created_at": datetime.now().isoformat(),
                "academic_project": "Casino Customer Segmentation Thesis",
                "ethics_ref": "10351-12382"
            }
    
    def register_model(self, model_type: str, model_path: str, 
                      metrics: Dict, notes: str = "") -> str:
        """
        Register a new model version.
        
        Args:
            model_type: 'segmentation' or 'promotion'
            model_path: Path where model is saved
            metrics: Performance metrics
            notes: Additional notes
            
        Returns:
            Model ID
        """
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_entry = {
            "model_id": model_id,
            "model_type": model_type,
            "model_path": model_path,
            "registered_at": datetime.now().isoformat(),
            "metrics": metrics,
            "notes": notes,
            "status": "active"
        }
        
        self.registry["models"].append(model_entry)
        self._save_registry()
        
        logger.info(f"Model registered: {model_id}")
        return model_id
    
    def get_model(self, model_id: str) -> Optional[Dict]:
        """Retrieve model information by ID"""
        for model in self.registry["models"]:
            if model["model_id"] == model_id:
                return model
        return None
    
    def get_best_model(self, model_type: str, metric: str = "roc_auc") -> Optional[Dict]:
        """Get best performing model by metric"""
        models = [m for m in self.registry["models"] 
                 if m["model_type"] == model_type and m["status"] == "active"]
        
        if not models:
            return None
        
        return max(models, key=lambda m: m["metrics"].get(metric, 0))
    
    def list_models(self, model_type: Optional[str] = None) -> pd.DataFrame:
        """List all registered models"""
        models = self.registry["models"]
        
        if model_type:
            models = [m for m in models if m["model_type"] == model_type]
        
        if not models:
            return pd.DataFrame()
        
        return pd.DataFrame(models)
    
    def deactivate_model(self, model_id: str):
        """Mark model as inactive"""
        for model in self.registry["models"]:
            if model["model_id"] == model_id:
                model["status"] = "inactive"
                model["deactivated_at"] = datetime.now().isoformat()
                break
        
        self._save_registry()
        logger.info(f"Model deactivated: {model_id}")
    
    def _save_registry(self):
        """Save registry to file"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)