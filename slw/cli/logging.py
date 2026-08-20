"""Rank-aware, QE-like terminal messages and execution timings.

The public ``print_header`` and ``print_footer`` functions remain deliberately
small so retained callers can keep using them. Native stage handlers can use
``RunLogger`` and ``TimerBook`` for rank-safe, line-oriented progress output.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import TextIO

_VERBOSITY = {
    "quiet": 0,
    "normal": 1,
    "high": 2,
    "debug": 3,
}


def _timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _program_label(program: str) -> str:
    label = str(program).replace("\\", "/").rsplit("/", 1)[-1]
    if label.lower().endswith(".x"):
        label = label[:-2]
    return label.upper()


def format_duration(seconds: float) -> str:
    """Format a non-negative duration without terminal control codes."""
    value = max(0.0, float(seconds))
    hours, remainder = divmod(value, 3600.0)
    minutes, secs = divmod(remainder, 60.0)
    if hours >= 1.0:
        return f"{int(hours)}h{int(minutes):02d}m{secs:05.2f}s"
    if minutes >= 1.0:
        return f"{int(minutes)}m{secs:05.2f}s"
    return f"{secs:.2f}s"


def format_timing_summary(
    program: str,
    *,
    cpu_seconds: float,
    wall_seconds: float,
) -> str:
    """Return a QE-style one-line CPU/WALL timing summary."""
    return (
        f"{_program_label(program)} : CPU {format_duration(cpu_seconds)} "
        f"WALL {format_duration(wall_seconds)}"
    )


@dataclass(frozen=True)
class TimingRecord:
    """Accumulated timing for one named phase."""

    name: str
    calls: int
    cpu_seconds: float
    wall_seconds: float


@dataclass
class PhaseTiming:
    """Timing populated when a ``TimerBook.phase`` context exits."""

    name: str
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0


@dataclass
class _TimingAccumulator:
    calls: int = 0
    cpu_seconds: float = 0.0
    wall_seconds: float = 0.0


class TimerBook:
    """Collect named CPU and wall-clock phase timings.

    Repeated phase names are accumulated. Clock callables are injectable to
    keep tests deterministic and to make the class usable in other runtimes.
    """

    def __init__(
        self,
        *,
        wall_clock: Callable[[], float] = time.perf_counter,
        cpu_clock: Callable[[], float] = time.process_time,
    ) -> None:
        self._wall_clock = wall_clock
        self._cpu_clock = cpu_clock
        self._records: dict[str, _TimingAccumulator] = {}
        self._lock = threading.RLock()

    @contextmanager
    def phase(self, name: str) -> Iterator[PhaseTiming]:
        """Measure one phase and add it to the named accumulated record."""
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("timer phase name cannot be empty")
        sample = PhaseTiming(normalized)
        wall_start = self._wall_clock()
        cpu_start = self._cpu_clock()
        try:
            yield sample
        finally:
            sample.cpu_seconds = max(0.0, self._cpu_clock() - cpu_start)
            sample.wall_seconds = max(0.0, self._wall_clock() - wall_start)
            self.record(
                normalized,
                cpu_seconds=sample.cpu_seconds,
                wall_seconds=sample.wall_seconds,
            )

    def record(
        self,
        name: str,
        *,
        cpu_seconds: float,
        wall_seconds: float,
        calls: int = 1,
    ) -> None:
        """Add an externally measured phase, including MPI-reduced timings."""
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("timer record name cannot be empty")
        if calls < 0:
            raise ValueError("timer calls cannot be negative")
        cpu = float(cpu_seconds)
        wall = float(wall_seconds)
        if cpu < 0.0 or wall < 0.0:
            raise ValueError("timer durations cannot be negative")
        with self._lock:
            target = self._records.setdefault(normalized, _TimingAccumulator())
            target.calls += int(calls)
            target.cpu_seconds += cpu
            target.wall_seconds += wall

    def get(self, name: str) -> TimingRecord | None:
        """Return an immutable snapshot for ``name``, if present."""
        with self._lock:
            value = self._records.get(name)
            if value is None:
                return None
            return TimingRecord(
                name=name,
                calls=value.calls,
                cpu_seconds=value.cpu_seconds,
                wall_seconds=value.wall_seconds,
            )

    @property
    def records(self) -> tuple[TimingRecord, ...]:
        """Return insertion-ordered immutable snapshots of all phases."""
        with self._lock:
            return tuple(
                TimingRecord(
                    name=name,
                    calls=value.calls,
                    cpu_seconds=value.cpu_seconds,
                    wall_seconds=value.wall_seconds,
                )
                for name, value in self._records.items()
            )

    def aggregate(self, name: str = "total") -> TimingRecord | None:
        """Sum all recorded phases.

        This is intended for sequential, non-overlapping phases. A separately
        measured top-level ``total`` phase should be preferred when phases can
        overlap or nest.
        """
        records = self.records
        if not records:
            return None
        return TimingRecord(
            name=name,
            calls=sum(item.calls for item in records),
            cpu_seconds=sum(item.cpu_seconds for item in records),
            wall_seconds=sum(item.wall_seconds for item in records),
        )

    def summary_record(self, phase: str = "total") -> TimingRecord | None:
        """Use a named total when available, otherwise aggregate all phases."""
        return self.get(phase) or self.aggregate(name=phase)

    def summary_line(self, program: str, *, phase: str = "total") -> str | None:
        record = self.summary_record(phase)
        if record is None:
            return None
        return format_timing_summary(
            program,
            cpu_seconds=record.cpu_seconds,
            wall_seconds=record.wall_seconds,
        )


@dataclass(frozen=True)
class _ProgressState:
    completed: int
    emitted_at: float


class RunLogger:
    """Rank-aware, line-oriented logger for one SLW run.

    Normal output is emitted only by ``root_rank``. ``quiet`` suppresses
    informational messages but retains warnings, errors, timing summaries, and
    the completion footer. No method emits carriage returns or ANSI escapes.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        rank: int = 0,
        root_rank: int = 0,
        root_only: bool = True,
        verbosity: str = "normal",
        timers: TimerBook | None = None,
        progress_clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        normalized = str(verbosity).strip().lower()
        if normalized not in _VERBOSITY:
            choices = ", ".join(_VERBOSITY)
            raise ValueError(f"verbosity must be one of {choices}")
        self.stream = stream if stream is not None else sys.stdout
        self.rank = int(rank)
        self.root_rank = int(root_rank)
        self.root_only = bool(root_only)
        self.verbosity = normalized
        self.timers = timers if timers is not None else TimerBook()
        self._progress_clock = progress_clock
        self._progress: dict[str, _ProgressState] = {}
        self._write_lock = threading.RLock()

    @property
    def is_root(self) -> bool:
        return self.rank == self.root_rank

    def enabled(self, level: str = "normal") -> bool:
        normalized = str(level).strip().lower()
        if normalized not in _VERBOSITY:
            raise ValueError(f"unknown log level: {level}")
        if self.root_only and not self.is_root:
            return False
        return _VERBOSITY[self.verbosity] >= _VERBOSITY[normalized]

    def _write(self, message: str, *, flush: bool = True) -> None:
        if self.root_only and not self.is_root:
            return
        lines = str(message).splitlines() or [""]
        with self._write_lock:
            for line in lines:
                print(line, file=self.stream)
            if flush:
                self.stream.flush()

    def log(self, message: str, *, level: str = "normal") -> None:
        if self.enabled(level):
            self._write(f"     {message}")

    def info(self, message: str) -> None:
        self.log(message, level="normal")

    def high(self, message: str) -> None:
        self.log(message, level="high")

    def debug(self, message: str) -> None:
        self.log(message, level="debug")

    def warning(self, message: str) -> None:
        if not self.root_only or self.is_root:
            self._write(f"     WARNING: {message}")

    def error(self, message: str) -> None:
        if not self.root_only or self.is_root:
            self._write(f"     ERROR: {message}")

    def header(
        self,
        *,
        program: str,
        version: str,
        calculation: str,
        prefix: str,
        outdir: str,
        savedir: str,
        mpi_size: int,
    ) -> None:
        """Print the compatible run header on the root rank."""
        if self.root_only and not self.is_root:
            return
        print_header(
            self.stream,
            program=program,
            version=version,
            calculation=calculation,
            prefix=prefix,
            outdir=outdir,
            savedir=savedir,
            mpi_size=mpi_size,
        )

    def backend(
        self,
        backend: str,
        *,
        execution: str | None = None,
        input_format: str | None = None,
    ) -> None:
        """Emit one stable backend-description line at normal verbosity."""
        if self.enabled("normal"):
            self._write(
                format_backend_line(
                    backend,
                    execution=execution,
                    input_format=input_format,
                )
            )

    @contextmanager
    def phase(
        self,
        name: str,
        *,
        label: str | None = None,
        level: str = "normal",
    ) -> Iterator[PhaseTiming]:
        """Time a phase and emit start/completion lines at ``level``."""
        display = label or name
        self.log(f"{display} ...", level=level)
        sample: PhaseTiming | None = None
        failed = False
        try:
            with self.timers.phase(name) as current:
                sample = current
                yield current
        except BaseException:
            failed = True
            raise
        finally:
            if sample is not None:
                suffix = " FAILED" if failed else ""
                self.log(
                    f"{display}{suffix} : CPU {format_duration(sample.cpu_seconds)} "
                    f"WALL {format_duration(sample.wall_seconds)}",
                    level=level,
                )

    def progress(
        self,
        label: str,
        completed: int,
        total: int,
        *,
        every: int = 1,
        min_interval: float = 0.0,
        level: str = "normal",
        force: bool = False,
    ) -> bool:
        """Emit a throttled, newline-terminated ``completed/total`` update.

        The first update and the final update are always eligible. Intermediate
        updates must satisfy both the item-count and elapsed-time throttles.
        Returns whether a line was emitted.
        """
        done = int(completed)
        count = int(total)
        step = int(every)
        interval = float(min_interval)
        if done < 0 or count < 0 or done > count:
            raise ValueError("progress requires 0 <= completed <= total")
        if step < 1:
            raise ValueError("progress every must be at least 1")
        if interval < 0.0:
            raise ValueError("progress min_interval cannot be negative")

        key = str(label)
        now = self._progress_clock()
        previous = self._progress.get(key)
        final = done == count
        if previous is None:
            eligible = True
        elif done == previous.completed and not force:
            eligible = False
        else:
            count_ready = done - previous.completed >= step
            time_ready = now - previous.emitted_at >= interval
            eligible = force or final or (count_ready and time_ready)

        if not eligible or not self.enabled(level):
            return False

        percent = 100.0 if count == 0 else 100.0 * done / count
        self.log(
            f"{key}: completed {done}/{count} ({percent:.1f}%)",
            level=level,
        )
        self._progress[key] = _ProgressState(done, now)
        return True

    def reset_progress(self, label: str | None = None) -> None:
        if label is None:
            self._progress.clear()
        else:
            self._progress.pop(str(label), None)

    def timing_summary(self, program: str, *, phase: str = "total") -> bool:
        """Emit the selected timer summary and report whether it existed."""
        if self.root_only and not self.is_root:
            return False
        line = self.timers.summary_line(program, phase=phase)
        if line is None:
            return False
        self._write(line)
        return True

    def footer(self, program: str, *, timing_phase: str = "total") -> None:
        """Print an optional timing summary followed by the JOB DONE footer."""
        if self.root_only and not self.is_root:
            return
        print_footer(
            self.stream,
            program=program,
            timers=self.timers,
            timing_phase=timing_phase,
        )


def format_backend_line(
    backend: str,
    *,
    execution: str | None = None,
    input_format: str | None = None,
) -> str:
    """Format one backend line suitable for redirected MPI stdout."""
    details = []
    if execution:
        details.append(str(execution))
    if input_format:
        details.append(f"input_format={input_format}")
    suffix = f" [{', '.join(details)}]" if details else ""
    return f"     backend     = {backend}{suffix}"


def print_header(
    stream: TextIO,
    *,
    program: str,
    version: str,
    calculation: str,
    prefix: str,
    outdir: str,
    savedir: str,
    mpi_size: int,
) -> None:
    print(f"Program {program.upper()} v.{version} starts on {_timestamp()}", file=stream)
    print(file=stream)
    print(f"     calculation = {calculation}", file=stream)
    print(f"     prefix      = {prefix}", file=stream)
    print(f"     outdir      = {outdir}", file=stream)
    print(f"     savedir     = {savedir}", file=stream)
    print(f"     MPI ranks   = {mpi_size}", file=stream)
    print(file=stream, flush=True)


def print_footer(
    stream: TextIO,
    *,
    program: str,
    timers: TimerBook | None = None,
    timing_phase: str = "total",
    cpu_seconds: float | None = None,
    wall_seconds: float | None = None,
) -> None:
    """Print a compatible JOB DONE footer with an optional timing summary."""
    if timers is not None and (cpu_seconds is None or wall_seconds is None):
        record = timers.summary_record(timing_phase)
        if record is not None:
            if cpu_seconds is None:
                cpu_seconds = record.cpu_seconds
            if wall_seconds is None:
                wall_seconds = record.wall_seconds

    print(file=stream)
    if cpu_seconds is not None and wall_seconds is not None:
        print(
            format_timing_summary(
                program,
                cpu_seconds=cpu_seconds,
                wall_seconds=wall_seconds,
            ),
            file=stream,
        )
        print(file=stream)
    print(f"{program}: JOB DONE.", file=stream)
    print(f"Finished on {_timestamp()}", file=stream, flush=True)
