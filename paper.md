---
title: 'pyqt-formgen: Declarative Reactive Form Generation from Python Type Hints with Hierarchical State and Game-Engine Animation'
tags:
  - Python
  - PyQt6
  - forms
  - reactive
  - GUI
  - type hints
authors:
  - name: Tristan Simas
    orcid: 0000-0000-0000-0000  # [TODO: Add ORCID]
    affiliation: 1
affiliations:
  - name: McGill University
    index: 1
date: 13 January 2026
bibliography: paper.bib
---

# Summary

`pyqt-formgen` generates reactive PyQt6 forms from Python dataclass definitions. A configuration dataclass with type-annotated fields automatically becomes an interactive form with appropriate widgets—spinboxes for integers, checkboxes for booleans, comboboxes for enums, and recursively nested forms for dataclass-typed fields:

```python
@dataclass
class ProcessingConfig:
    input_path: str = ""
    num_workers: int = 4
    enable_gpu: bool = False
    advanced: Optional[AdvancedConfig] = None  # Checkbox-controlled nested form

form = ParameterFormManager(ProcessingConfig)
config = form.collect_values()  # Typed ProcessingConfig instance
```

The framework integrates with `objectstate` for hierarchical configuration inheritance—where a child configuration can inherit defaults from a parent scope while tracking which values were explicitly overridden. Widget placeholders dynamically display inherited values (e.g., "Pipeline default: 4"), and a game-engine-style flash animation system provides instant visual feedback when values change across windows.

# Statement of Need

Scientific applications require complex configuration interfaces. A microscopy processing pipeline might have global settings, per-experiment settings, per-sample settings, and per-step function arguments—each inheriting from their parent scope. Building these interfaces manually in PyQt requires repetitive boilerplate: creating widgets, connecting signals, managing value collection, and handling the bidirectional flow between UI state and data models.

Existing form generators address only part of this problem:

- **Qt Designer** provides visual layout but no type-based widget selection or runtime form generation
- **magicgui** [@magicgui] generates widgets from function signatures but targets napari integration, not hierarchical forms
- **PyQt-Fluent-Widgets** provides styled components but no form generation or state management

`pyqt-formgen` differs in four key areas:

1. **Discriminated union type dispatch**: Widget creation uses metaclass-registered `ParameterInfo` types (`OptionalDataclassInfo`, `DirectDataclassInfo`, `GenericInfo`) that auto-select based on type introspection, eliminating if-else chains
2. **ObjectState integration**: Forms bind to hierarchical state objects that track inheritance, provide placeholder resolution, and support git-style undo/redo with branches
3. **Cross-window synchronization**: Changes in one window immediately update placeholder text in all related windows without explicit save operations
4. **O(1) flash animations**: A global coordinator batches animation updates, while per-window overlays render all effects in a single paint call—achieving 144Hz performance regardless of element count

# Software Design

## Widget Protocol ABCs

The framework defines six ABC contracts that normalize Qt's inconsistent widget APIs:

| ABC | Purpose |
|-----|---------|
| `ValueGettable` | `get_value() -> Any` for all widgets |
| `ValueSettable` | `set_value(value)` for all widgets |
| `PlaceholderCapable` | `set_placeholder(text)` for inheritance display |
| `RangeConfigurable` | `configure_range(min, max)` for numeric widgets |
| `EnumSelectable` | `set_enum_options(type)` for dropdown population |
| `ChangeSignalEmitter` | `connect_change_signal(callback)` unifying Qt signals |

Adapter classes (`LineEditAdapter`, `SpinBoxAdapter`, `ComboBoxAdapter`, `CheckBoxAdapter`, `CheckboxGroupAdapter`) wrap Qt widgets to implement these contracts. A `PyQtWidgetMeta` metaclass combines Qt's metaclass with `ABCMeta` for proper multiple inheritance.

## Type-Based Widget Dispatch

Parameter types use discriminated unions with metaclass auto-registration. Rather than boolean flags, polymorphic `ParameterInfo` types auto-select based on type introspection:

```python
@dataclass
class OptionalDataclassInfo(ParameterInfoBase, metaclass=ParameterInfoMeta):
    widget_creation_type: str = "OPTIONAL_NESTED"

    @staticmethod
    def matches(param_type: Type) -> bool:
        is_optional = get_origin(param_type) is Union and type(None) in get_args(param_type)
        inner_type = next(arg for arg in get_args(param_type) if arg is not type(None))
        return is_optional and is_dataclass(inner_type)
```

Services dispatch to handlers named by class (`_reset_OptionalDataclassInfo`, `_collect_DirectDataclassInfo`), enabling type-safe handling without explicit dispatch tables.

## Abstract UI Components

The framework provides reusable ABC bases for common UI patterns:

**AbstractManagerWidget**: Template-method base for CRUD list managers. Declarative configuration via class attributes (`BUTTON_CONFIGS`, `ITEM_HOOKS`, `PREVIEW_FIELD_CONFIGS`) defines toolbar buttons, item accessors, and preview fields. Subclasses implement only domain-specific hooks (`action_add`, `_show_item_editor`).

**AbstractTableBrowser[T]**: Generic base for searchable table browsers with configurable columns, `SearchService` integration, and selection handling. Subclasses define `get_columns()`, `extract_row_data()`, and `get_searchable_text()`.

**Mixins**: Composable behaviors via mixin classes:
- `FlashMixin`: Provides `queue_flash(key)` API for visual feedback
- `CrossWindowPreviewMixin`: Subscribes to `ObjectStateRegistry` changes with debounced refresh
- `ScrollableFormMixin`: Scroll-to-field navigation for large forms

## Protocol-Based Dependency Injection

The framework uses Protocol classes for application-specific integrations, avoiding hard dependencies:

| Protocol | Purpose | Registration |
|----------|---------|--------------|
| `FunctionRegistryProtocol` | Function lookup by name | `register_function_registry(impl)` |
| `LLMServiceProtocol` | Code generation via LLM | `register_llm_service(impl)` |
| `CodegenProvider` | Config-to-code serialization | `register_codegen_provider(impl)` |
| `LogDiscoveryProvider` | Log file discovery | `register_log_discovery_provider(impl)` |
| `PreviewFormatterRegistry` | Custom preview formatting | `register_preview_formatter(type, fn)` |

Applications register implementations at startup; the framework calls protocol methods without knowing concrete types.

## Flash Animation Architecture

The animation system uses game-engine patterns for O(1) rendering per window:

1. **GlobalFlashCoordinator** (singleton): One 60fps (or 144fps) timer for all windows pre-computes all colors in a single pass
2. **WindowFlashOverlay** (per-window): Renders all flash rectangles in one `paintEvent` using cached geometry
3. **FlashMixin** (per-widget): API for registering elements and triggering scope-keyed flashes

This architecture scales with the number of *animating* elements, not total elements. An experimental OpenGL backend (`WindowFlashOverlayGL`) uses instanced rendering for workloads with many simultaneous flashes.

## Service Layer

Cross-cutting concerns are factored into stateless services:

- **FieldChangeDispatcher**: Singleton routing all field changes with reentrancy guards
- **ParameterOpsService**: Type-safe reset and placeholder refresh with auto-discovered handlers
- **ValueCollectionService**: Nested value collection with discriminated union dispatch
- **SignalService**: Context managers for signal blocking and cross-window registration
- **WindowManager**: Scoped window registry ensuring one window per `scope_id` with navigation
- **ScopeColorService**: Consistent color generation for scope-based visual grouping
- **ScopeTokenService**: Unique token generation for stable scope identification

# Research Impact Statement

`pyqt-formgen` is the UI framework for OpenHCS, an open-source platform for high-content screening microscopy. In OpenHCS:

- **Pipeline configuration** uses hierarchical forms where global settings inherit to per-plate settings, which inherit to per-step function arguments
- **Function editors** dynamically generate forms from callable signatures, allowing any Python function to become a pipeline step
- **Live preview** updates step labels and indicators across windows as users edit parameters in any scope
- **Time-travel debugging** uses `ObjectState`'s git-style history to undo entire UI actions including adding/removing pipeline steps

The framework processes configuration hierarchies with 50+ nested dataclass fields and 20+ simultaneously open windows while maintaining <16ms animation frame times. This responsiveness is critical for interactive scientific workflows where researchers iterate rapidly on analysis parameters.

The scope-based visual feedback system—where each pipeline step gets a consistent color across its editor window, list item, and flash animations—provides spatial memory cues that reduce cognitive load in complex pipelines with many configuration levels.

# State of the Field

Form generation from type hints exists in several ecosystems:

**magicgui** [@magicgui] creates widgets from function signatures and integrates with napari's plugin system. It focuses on single-function forms rather than hierarchical configuration.

**pydantic-settings** [@pydantic] validates configuration at load time but provides no UI generation.

**Qt Model/View** provides data binding but requires manual model/delegate implementation per field type.

**WxPython FormBuilder** generates forms from XML definitions, not runtime type introspection.

`pyqt-formgen` uniquely combines runtime type-based generation, hierarchical state with placeholder inheritance, cross-window reactivity, and high-performance animation—addressing the complete lifecycle from dataclass definition to interactive, multi-window configuration editing.

# AI Usage Disclosure

Generative AI (Claude) assisted with code generation and documentation. All content was reviewed, tested, and integrated by human developers. Core architectural decisions—discriminated union dispatch, game-engine animation architecture, service layer factoring, ObjectState integration—were human-designed based on performance requirements and PyQt6 best practices.

# Acknowledgements

This work was supported by [TODO: Add funding sources].

# References

