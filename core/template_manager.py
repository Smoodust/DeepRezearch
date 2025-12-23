from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from typing import Dict
from loguru import logger
import os

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class TemplateManager(object, metaclass=Singleton):
    """Управляет загрузкой и кешированием шаблонов"""
    
    def __init__(self, templates_dir: str | None = "prompts"):
        self.templates_dir = templates_dir
        self._templates_cache: Dict[str, Template] = {}
        
        if templates_dir and os.path.isdir(templates_dir):
            self.jinja_env = Environment(
                loader=FileSystemLoader(templates_dir),
                autoescape=select_autoescape(enabled_extensions=()),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            self.jinja_env = None
    
    def load_template(self, name: str) -> Template:
        """Загружает шаблон с кешированием"""
        if name not in self._templates_cache:
            if not self.jinja_env:
                raise RuntimeError("Jinja environment not initialized")
            
            try:
                self._templates_cache[name] = self.jinja_env.get_template(name)
                logger.debug(f"Template loaded: {name}")
            except Exception as e:
                logger.error(f"Failed to load template {name}: {e}")
                raise
        
        return self._templates_cache[name]
    
    def render_template(self, name: str, **kwargs) -> str:
        """Загружает и рендерит шаблон одновременно"""
        template = self.load_template(name)
        return template.render(**kwargs)