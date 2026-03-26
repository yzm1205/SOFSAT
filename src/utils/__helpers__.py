from pathlib import Path
from typing import Union, Optional


def delete_file(file_pt: Path) -> None:
    try:
        file_pt.unlink()
    except FileNotFoundError:
        pass


def full_path(inp_dir_or_path: str) -> Path:
    """Returns full path"""
    return Path(inp_dir_or_path).expanduser().resolve()


def mkdir_p(inp_dir_or_path: Union[str, Path]) -> Path:
    """Give a file/dir path, makes sure that all the directories exists"""
    inp_dir_or_path = full_path(inp_dir_or_path)
    if inp_dir_or_path.suffix:  # file
        inp_dir_or_path.parent.mkdir(parents=True, exist_ok=True)
    else:  # dir
        inp_dir_or_path.mkdir(parents=True, exist_ok=True)
    return inp_dir_or_path


def derive_model_label(model_name: Optional[str], custom_class_path: Optional[str]) -> str:
    if model_name:
        return model_name.split("/")[-1]
    if custom_class_path:
        return custom_class_path.split(".")[-1]
    return "custom_embedder"