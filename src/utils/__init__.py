"""
Módulo de Utilidades
====================

Funciones compartidas para evitar duplicación de código.
"""

from .common import (
    load_model,
    load_data,
    print_header,
    print_section,
    print_metric,
    print_summary,
    safe_execute,
    format_percentage,
    format_currency,
    create_directory,
    get_project_root,
    add_project_to_path,
    EMOJI_SUCCESS,
    EMOJI_ERROR,
    EMOJI_WARNING,
    EMOJI_INFO
)

__all__ = [
    'load_model',
    'load_data',
    'print_header',
    'print_section',
    'print_metric',
    'print_summary',
    'safe_execute',
    'format_percentage',
    'format_currency',
    'create_directory',
    'get_project_root',
    'add_project_to_path',
    'EMOJI_SUCCESS',
    'EMOJI_ERROR',
    'EMOJI_WARNING',
    'EMOJI_INFO'
]
