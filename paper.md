---
title: 'pyqt-formgen: Reactive Form Generation from Python Type Hints for PyQt6'
tags:
  - Python
  - PyQt6
  - forms
  - reactive
  - GUI
  - type hints
authors:
  - name: Tristan Simas
    orcid: 0000-0002-6526-3149
    affiliation: 1
affiliations:
  - name: McGill University
    index: 1
date: 13 January 2026
bibliography: paper.bib
---

# Summary

`pyqt-formgen` generates reactive PyQt6 forms from Python objects with inspectable signatures—dataclasses, regular classes, functions, or callables with type hints. Type-annotated parameters automatically become appropriate widgets with bidirectional data binding:

```python
@dataclass
class Config:
    path: str = ""
    workers: int = 4
    gpu: bool = False
    advanced: Optional[AdvancedConfig] = None  # Nested form

form = ParameterFormManager(Config)  # Auto-generates UI
config = form.collect_values()       # Returns typed instance
```

The framework uses `UnifiedParameterAnalyzer` to introspect any target—analyzing function signatures, dataclass fields, class `__init__` methods, or `__call__` implementations through a single code path. Forms support hierarchical inheritance via `objectstate` [@objectstate] integration, where child configurations inherit parent defaults with placeholder text showing inherited values (e.g., "Pipeline default: 4").

# Statement of Need

Desktop applications in PyQt require substantial boilerplate: creating widgets, connecting signals, collecting values, and synchronizing UI state with data models. This burden multiplies for applications with nested configuration—scientific pipelines, CAD tools, or any domain requiring hierarchical settings.

Existing solutions address only fragments:

- **Qt Designer**: Visual layout, but no type-based widget selection or runtime generation
- **magicgui** [@magicgui]: Function signature widgets for napari, but no hierarchical forms or cross-window sync
- **pydantic-settings** [@pydantic]: Validation without UI generation

`pyqt-formgen` brings React-style declarative UI to PyQt6, eliminating manual widget management while providing:

1. **Universal introspection**: Single analyzer handles functions, dataclasses, classes, and callables
2. **Hierarchical state**: Placeholder inheritance, scope tracking, git-style undo via ObjectState
3. **Cross-window reactivity**: Changes propagate to all related windows without explicit saves
4. **O(1) animations**: Game-engine-style rendering scales with animating elements, not total widgets

# Software Design

## Widget Protocol ABCs

Six ABC contracts normalize Qt's inconsistent widget APIs: `ValueGettable`, `ValueSettable`, `PlaceholderCapable`, `RangeConfigurable`, `EnumSelectable`, and `ChangeSignalEmitter`. Adapter classes wrap Qt widgets (`LineEditAdapter`, `SpinBoxAdapter`, etc.) to implement these contracts, enabling polymorphic widget handling. A `PyQtWidgetMeta` metaclass combines Qt's metaclass with `ABCMeta` for proper multiple inheritance.

## Discriminated Union Type Dispatch

Parameter types use metaclass-registered discriminated unions. `ParameterInfo` subclasses (`OptionalDataclassInfo`, `DirectDataclassInfo`, `GenericInfo`) define `matches()` predicates; the factory selects the first match:

```python
class OptionalDataclassInfo(ParameterInfoBase, metaclass=ParameterInfoMeta):
    @staticmethod
    def matches(param_type: Type) -> bool:
        return is_optional(param_type) and is_dataclass(get_inner_type(param_type))
```

Services dispatch to handlers by class name (`_reset_OptionalDataclassInfo`), enabling type-safe operations without dispatch tables.

## Abstract UI Components

**AbstractManagerWidget**: Template-method base for CRUD list managers with declarative configuration (`BUTTON_CONFIGS`, `PREVIEW_FIELD_CONFIGS`). Subclasses implement only domain hooks.

**AbstractTableBrowser[T]**: Generic searchable table with `SearchService` integration.

**Mixins**: `FlashMixin` (visual feedback), `CrossWindowPreviewMixin` (debounced refresh on state changes), `ScrollableFormMixin` (scroll-to-field navigation).

## Protocol-Based Extensibility

Protocol classes enable application-specific integrations without hard dependencies: `FunctionRegistryProtocol`, `LLMServiceProtocol`, `CodegenProvider`, `PreviewFormatterRegistry`. Applications register implementations at startup; the framework calls protocol methods without knowing concrete types.

## Flash Animation Architecture

Game-engine patterns achieve O(1) per-window rendering:

1. **GlobalFlashCoordinator**: Single timer pre-computes all colors
2. **WindowFlashOverlay**: Renders all rectangles in one `paintEvent`
3. **FlashMixin**: Per-widget API for scope-keyed flashes

Performance scales with animating elements, not total widgets. Optional OpenGL backend uses instanced rendering.

## Service Layer

Stateless services handle cross-cutting concerns: `FieldChangeDispatcher` (change routing with reentrancy guards), `ParameterOpsService` (type-safe reset/refresh), `ValueCollectionService` (nested collection), `WindowManager` (scoped window registry), `ScopeColorService` (consistent scope colors).

# Research Application

`pyqt-formgen` powers OpenHCS, an open-source high-content screening platform. Pipeline configuration uses hierarchical forms—global settings inherit to per-experiment, per-sample, and per-step scopes. Function editors generate forms from callable signatures, making any Python function a pipeline step. The framework handles 50+ nested fields across 20+ windows with responsive updates during active editing.

# AI Usage Disclosure

Generative AI (Claude) assisted with code generation and documentation. All content was reviewed, tested, and integrated by human developers. Core architectural decisions—discriminated union dispatch, game-engine animation, ABC protocol system, ObjectState integration—were human-designed.

# Acknowledgements

This work was supported by [TODO: Add funding sources].

# References
