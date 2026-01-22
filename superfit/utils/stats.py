import time
import numpy as np
from typing import Dict, Any, List, Tuple
from contextlib import contextmanager
from .logger import logger


def _flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dictionary with dot-separated keys."""
    result = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            if value.get("GLFunction", False):
                result[full_key] = value
            else:
                result.update(_flatten_dict(value, full_key))
        else:
            result[full_key] = value
    return result


class StatisticsManager:
    """Statistics manager with path-based scoping and context managers."""
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._scope_stack: List[str] = []
    
    def _get_current_scope_path(self) -> str:
        """Get the current scope path as a dot-separated string."""
        return ".".join(self._scope_stack) if self._scope_stack else ""
    
    def _resolve_path(self, path: str, root: bool = False) -> List[str]:
        """Resolve a path string to a list of path components."""
        if root or (path.startswith("/") and len(path) > 1):
            path = path.lstrip("/")
            return path.split(".") if path else []
        
        current_scope = self._get_current_scope_path()
        full_path = f"{current_scope}.{path}" if current_scope and path else (current_scope or path)
        return full_path.split(".") if full_path else []
    
    def _get_nested_dict(self, path_components: List[str], create: bool = False) -> Dict[str, Any]:
        """Navigate to a nested dict location, creating intermediate dicts if needed."""
        current = self._data
        for component in path_components:
            if not isinstance(current, dict):
                if create:
                    current = {}
                else:
                    raise KeyError(f"Path component '{component}' is not a dict")
            
            if component not in current:
                if create:
                    current[component] = {}
                else:
                    raise KeyError(f"Path component '{component}' not found")
            
            current = current[component]
        
        return current
    
    def _get_value(self, path_components: List[str], check_exists: bool = False) -> Tuple[Any, bool]:
        """Get a value at a path, optionally checking if path exists.
        
        Returns:
            (value, exists) tuple where exists indicates if the path exists
        """
        if not path_components:
            return self._data, True
        
        current = self._data
        for component in path_components[:-1]:
            if not isinstance(current, dict) or component not in current:
                return None, False
            current = current[component]
        
        if not isinstance(current, dict):
            return None, False
        
        key = path_components[-1]
        exists = key in current
        return current.get(key), exists
    
    @contextmanager
    def scope(self, path: str):
        """Context manager for setting a scope prefix."""
        path_components = path.split(".")
        self._scope_stack.extend(path_components)
        try:
            yield
        finally:
            for _ in path_components:
                if self._scope_stack:
                    self._scope_stack.pop()
    
    def record(self, path: str, value: Any, log: bool = True, as_list=False) -> None:
        """Record a statistic value at a path, optionally logging it."""
        path_components = self._resolve_path(path)
        if not path_components:
            return
        
        parent_path, key = path_components[:-1], path_components[-1]
        parent_dict = self._get_nested_dict(parent_path, create=True) if parent_path else self._data
        if as_list:
            if key not in parent_dict:
                parent_dict[key] = []
            parent_dict[key].append(value)
        else:
            parent_dict[key] = value
        
        if log:
            if as_list:
                formatted_value = np.mean(parent_dict[key])
                logger.info(f"Mean {path}: {formatted_value:.6f}")
            else:
                formatted_value = f"{value:.6f}" if isinstance(value, float) else value
                logger.info(f"{path}: {formatted_value}")
    
    def get(self, path: str = "", default: Any = None, root: bool = False) -> Any:
        """Get a value or dict at a path, returning default if path doesn't exist."""
        if path == "" and not root:
            current_scope = self._get_current_scope_path()
            if current_scope:
                try:
                    return self._get_nested_dict(current_scope.split("."), create=False)
                except KeyError:
                    return default
            return self._data
        
        path_components = self._resolve_path(path, root=root)
        if not path_components:
            return self._data
        
        value, exists = self._get_value(path_components, check_exists=True)
        return default if not exists else value
    
    @contextmanager
    def timer(self, name: str, log: bool = True):
        """Context manager for timing operations."""
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.record(f"{name}.time", elapsed, log=False)
            if log:
                logger.info(f"==================== {name}: {elapsed:.3f} seconds ====================")
    
    def get_current_scope_dict(self) -> Dict[str, Any]:
        """Get all records in the current scope as a flat dictionary with dot-separated keys."""
        current_scope = self._get_current_scope_path()
        
        if not current_scope:
            return self.get_dict()
        
        try:
            scope_dict = self._get_nested_dict(current_scope.split("."), create=False)
            return _flatten_dict(scope_dict, current_scope)
        except KeyError:
            return {}
    
    def get_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary with dot-separated keys."""
        return _flatten_dict(self._data)
    
    def reset(self) -> None:
        """Reset all statistics."""
        self._data.clear()
        self._scope_stack.clear()


# Global singleton instance
Stats = StatisticsManager()
