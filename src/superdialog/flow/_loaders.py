import json
from pathlib import Path
from typing import Type, TypeVar, Union

import yaml
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def load_from_yaml_file(model_cls: Type[T], file_path: Union[str, Path]) -> T:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Flow file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {file_path}: {e}")

    try:
        return model_cls.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid flow definition in {file_path}: {e}")


def load_from_yaml_string(model_cls: Type[T], yaml_string: str) -> T:
    try:
        data = yaml.safe_load(yaml_string)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML content: {e}")

    try:
        return model_cls.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid flow definition: {e}")


def load_from_json_file(model_cls: Type[T], file_path: Union[str, Path]) -> T:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Flow file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")

    # Keep parity with load_from_json_string(): file-based JSON can also be a
    # React Flow export and needs normalization before validation.
    if hasattr(model_cls, "_is_react_flow_format") and model_cls._is_react_flow_format(
        data
    ):
        data = model_cls._normalize_react_flow(data)

    try:
        return model_cls.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid flow definition in {file_path}: {e}")


def load_from_json_string(model_cls: Type[T], json_string: str) -> T:
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON content: {e}")

    # Auto-detect and transform React Flow format
    if hasattr(model_cls, "_is_react_flow_format") and model_cls._is_react_flow_format(
        data
    ):
        data = model_cls._normalize_react_flow(data)

    try:
        return model_cls.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Invalid flow definition: {e}")


def load_from_file(model_cls: Type[T], file_path: Union[str, Path]) -> T:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Flow file not found: {file_path}")

    extension = file_path.suffix.lower()

    if extension in [".yaml", ".yml"]:
        return load_from_yaml_file(model_cls, file_path)
    elif extension == ".json":
        return load_from_json_file(model_cls, file_path)
    else:
        raise ValueError(
            f"Unsupported file extension '{extension}'. "
            "Supported formats: .yaml, .yml, .json"
        )
