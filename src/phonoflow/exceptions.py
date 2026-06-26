"""Custom exceptions used by PhonoFlow."""


class PhonoFlowError(Exception):
    """Base class for user-facing PhonoFlow errors."""


class ConfigError(PhonoFlowError):
    """Raised when workflow configuration is invalid."""


class StructureReadError(PhonoFlowError):
    """Raised when a structure cannot be read."""


class StructureWriteError(PhonoFlowError):
    """Raised when a structure cannot be written."""


class BackendUnavailableError(PhonoFlowError):
    """Raised when a requested calculator backend is unavailable."""


class WorkflowError(PhonoFlowError):
    """Raised when a workflow step fails."""
