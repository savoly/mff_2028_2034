import ast
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal


Mode = Literal["name", "signature", "body"]


def find_duplicate_functions_in_notebooks(
    root: str | Path = ".",
    mode: Mode = "name",
    recursive: bool = False,
) -> dict[str, list[tuple[str, int, str]]]:
    """
    Find duplicated function definitions in Jupyter notebooks.

    Parameters
    ----------
    root
        Directory containing notebooks.
    mode
        How to compare functions:
        - "name": same function name
        - "signature": same full function signature
        - "body": same function body
    recursive
        Whether to search recursively.

    Returns
    -------
    dict
        Mapping from comparison key to a list of:
        (notebook filename, cell index, function name)
        Only duplicate entries are returned.
    """
    root = Path(root)
    notebook_paths = root.rglob("*.ipynb") if recursive else root.glob("*.ipynb")

    seen: defaultdict[str, list[tuple[str, int, str]]] = defaultdict(list)

    for path in notebook_paths:
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Skipping {path}: could not read JSON ({exc})")
            continue

        cells = nb.get("cells", [])
        if not isinstance(cells, list):
            print(f"Skipping {path}: invalid notebook format ('cells' is not a list)")
            continue

        for cell_idx, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue

            source = "".join(cell.get("source", []))
            if not source.strip():
                continue

            try:
                tree = ast.parse(source)
            except SyntaxError:
                # Skip cells that are not valid standalone Python
                continue

            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = _make_function_key(node, mode)
                    seen[key].append((path.name, cell_idx, node.name))

    return {k: v for k, v in seen.items() if len(v) > 1}


def _make_function_key(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    mode: Mode,
) -> str:
    if mode == "name":
        return node.name

    if mode == "signature":
        return _function_signature(node)

    if mode == "body":
        return _normalized_function_body(node)

    raise ValueError(f"Unsupported mode: {mode!r}")


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    kind = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args = ast.unparse(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{kind} {node.name}({args}){returns}"


def _normalized_function_body(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    # Compare full function implementation, but ignore the function name
    # so renamed copies still count as duplicates.
    cloned = _clone_function_without_name(node)
    return ast.dump(cloned, annotate_fields=True, include_attributes=False)


def _clone_function_without_name(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    if isinstance(node, ast.AsyncFunctionDef):
        return ast.AsyncFunctionDef(
            name="__FUNC__",
            args=node.args,
            body=node.body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=node.type_comment,
        )
    return ast.FunctionDef(
        name="__FUNC__",
        args=node.args,
        body=node.body,
        decorator_list=node.decorator_list,
        returns=node.returns,
        type_comment=node.type_comment,
    )


def print_duplicate_functions(
    duplicates: dict[str, list[tuple[str, int, str]]],
    show_key: bool = False,
) -> None:
    if not duplicates:
        print("No duplicates found.")
        return

    for key, locations in duplicates.items():
        print()
        if show_key:
            print(f"Key: {key}")
        else:
            print(f"Duplicate: {locations[0][2]}")

        for file_name, cell_idx, func_name in locations:
            print(f"  {func_name} in {file_name} (cell {cell_idx})")


# # same names
# dups = find_duplicate_functions_in_notebooks(mode="name")
# print_duplicate_functions(dups)
#
# same signatures
dups = find_duplicate_functions_in_notebooks(mode="signature")
print_duplicate_functions(dups, show_key=True)

# # same bodies, even if renamed
# dups = find_duplicate_functions_in_notebooks(mode="body")
# print_duplicate_functions(dups)
