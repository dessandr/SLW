"""Validated bridge from namelist values to retained argparse drivers."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import io
import shlex
import sys
import threading
from collections.abc import Mapping, Sequence
from difflib import get_close_matches
from typing import Any


class ArgumentAdapterError(ValueError):
    """Raised when namelist values do not match a retained driver parser."""


class _ParserCaptured(BaseException):
    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser


_PARSER_CAPTURE_LOCK = threading.RLock()


def _callable_without_required_arguments(function: Any) -> bool:
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return False
    return all(
        parameter.default is not inspect.Parameter.empty
        or parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }
        for parameter in signature.parameters.values()
    )


def parser_for(module_name: str) -> argparse.ArgumentParser:
    """Obtain a module's parser without executing its calculation."""
    module = importlib.import_module(module_name)
    for builder_name in ("build_arg_parser", "build_parser"):
        builder = getattr(module, builder_name, None)
        if callable(builder) and _callable_without_required_arguments(builder):
            parser = builder()
            if isinstance(parser, argparse.ArgumentParser):
                return parser

    main = getattr(module, "main", None)
    if not callable(main):
        raise ArgumentAdapterError(f"{module_name} has no callable main()")

    def capture(parser: argparse.ArgumentParser, *args: Any, **kwargs: Any) -> Any:
        raise _ParserCaptured(parser)

    with _PARSER_CAPTURE_LOCK:
        original = argparse.ArgumentParser.parse_args
        argparse.ArgumentParser.parse_args = capture
        try:
            try:
                main()
            except _ParserCaptured as found:
                return found.parser
        finally:
            argparse.ArgumentParser.parse_args = original
    raise ArgumentAdapterError(
        f"could not capture an ArgumentParser from {module_name}.main()"
    )


def _normalize(name: str) -> str:
    return name.lstrip("-").strip().lower().replace("-", "_")


def _known_names(parser: argparse.ArgumentParser) -> list[str]:
    names = []
    for action in parser._actions:
        if action.dest and action.dest != argparse.SUPPRESS:
            names.append(str(action.dest))
        names.extend(option.lstrip("-") for option in action.option_strings)
    return sorted(set(names), key=str.lower)


def _find_argument_action(
    parser: argparse.ArgumentParser,
    key: str,
) -> tuple[argparse.Action, str, str | None]:
    wanted = _normalize(key)
    exact: list[tuple[argparse.Action, str]] = []
    by_dest: list[argparse.Action] = []
    for action in parser._actions:
        for option in action.option_strings:
            if _normalize(option) == wanted:
                exact.append((action, option))
        if _normalize(str(action.dest)) == wanted:
            by_dest.append(action)
    if exact:
        action, option = exact[0]
        return action, "option", option
    if by_dest:
        return by_dest[0], "dest", None

    known = _known_names(parser)
    close = get_close_matches(key, known, n=3, cutoff=0.55)
    hint = f"; close matches: {', '.join(close)}" if close else ""
    raise ArgumentAdapterError(f"unknown input parameter {key!r}{hint}")


def _token(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, complex):
        return f"({value.real},{value.imag})"
    return str(value)


def _as_items(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return [value]


def _negative_option(action: argparse.Action) -> str | None:
    for option in action.option_strings:
        if _normalize(option).startswith("no_"):
            return option
    return None


def _positive_option(action: argparse.Action) -> str | None:
    for option in action.option_strings:
        if not _normalize(option).startswith("no_"):
            return option
    return action.option_strings[0] if action.option_strings else None


def _flag_tokens(
    parser: argparse.ArgumentParser,
    action: argparse.Action,
    *,
    key: str,
    value: Any,
    match_kind: str,
    exact_option: str | None,
) -> list[str]:
    if not isinstance(value, bool):
        raise ArgumentAdapterError(
            f"parameter {key!r} is a switch and requires .true. or .false."
        )

    if isinstance(action, argparse.BooleanOptionalAction):
        if match_kind == "option" and exact_option is not None:
            exact_is_negative = _normalize(exact_option).startswith("no_")
            if exact_is_negative:
                return [exact_option] if value else []
        option = _positive_option(action) if value else _negative_option(action)
        if option is None:
            raise ArgumentAdapterError(f"cannot represent {key}={value!r}")
        return [option]

    if match_kind == "option":
        if value:
            return [str(exact_option)]
        if _normalize(str(action.dest)) == _normalize(key):
            candidates = [item for item in parser._actions if item.dest == action.dest]
            for candidate in candidates:
                if isinstance(candidate, argparse._StoreFalseAction):
                    option = _positive_option(candidate)
                    return [option] if option is not None else []
        return []

    candidates = [item for item in parser._actions if item.dest == action.dest]
    for candidate in candidates:
        if isinstance(
            candidate, (argparse._StoreTrueAction, argparse._StoreFalseAction)
        ) and bool(candidate.const) == value:
            option = _positive_option(candidate)
            return [option] if option is not None else []
    if bool(action.default) == value:
        return []
    raise ArgumentAdapterError(f"cannot represent {key}={value!r} with the legacy parser")


def arguments_from_parameters(
    parser: argparse.ArgumentParser,
    parameters: Mapping[str, Any],
) -> list[str]:
    """Convert typed namelist parameters and validate them with argparse."""
    option_tokens: list[str] = []
    positional_tokens: list[str] = []
    raw_tokens: list[str] = []

    for key, value in parameters.items():
        if key in {"cli_args", "legacy_args"}:
            if isinstance(value, str):
                raw_tokens.extend(shlex.split(value))
            else:
                raw_tokens.extend(_token(item) for item in _as_items(value))
            continue
        if value is None:
            continue

        action, match_kind, exact_option = _find_argument_action(parser, key)
        is_flag = isinstance(
            action,
            (
                argparse._StoreTrueAction,
                argparse._StoreFalseAction,
                argparse.BooleanOptionalAction,
            ),
        )
        if is_flag:
            option_tokens.extend(
                _flag_tokens(
                    parser,
                    action,
                    key=key,
                    value=value,
                    match_kind=match_kind,
                    exact_option=exact_option,
                )
            )
            continue

        items = _as_items(value)
        if action.option_strings:
            option = exact_option or _positive_option(action)
            if option is None:
                raise ArgumentAdapterError(f"no option string found for {key!r}")
            if isinstance(action, argparse._AppendAction) and len(items) > 1 and action.nargs in (None, 1):
                for item in items:
                    option_tokens.extend([option, _token(item)])
            else:
                option_tokens.append(option)
                option_tokens.extend(_token(item) for item in items)
        else:
            positional_tokens.extend(_token(item) for item in items)

    argv = [*option_tokens, *positional_tokens, *raw_tokens]
    error_stream = io.StringIO()
    try:
        with contextlib.redirect_stderr(error_stream):
            parser.parse_args(argv)
    except SystemExit as exc:
        detail = error_stream.getvalue().strip()
        raise ArgumentAdapterError(detail or f"invalid legacy arguments: {argv}") from exc
    return argv


@contextlib.contextmanager
def _temporary_argv(program: str, argv: Sequence[str]):
    previous = sys.argv
    sys.argv = [program, *argv]
    try:
        yield
    finally:
        sys.argv = previous


def invoke(module_name: str, argv: Sequence[str], *, program: str) -> int:
    """Invoke a retained driver in-process while preserving ``sys.argv``."""
    module = importlib.import_module(module_name)
    main = getattr(module, "main", None)
    if not callable(main):
        raise ArgumentAdapterError(f"{module_name} has no callable main()")

    signature = inspect.signature(main)
    accepts_argv = "argv" in signature.parameters
    try:
        with _temporary_argv(program, argv):
            result = main(list(argv)) if accepts_argv else main()
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return int(exc.code)
        raise ArgumentAdapterError(str(exc.code)) from exc
    return int(result) if isinstance(result, int) else 0
