import io
import unittest

from slw.cli.logging import (
    RunLogger,
    TimerBook,
    format_backend_line,
    print_footer,
)


class LoggingTests(unittest.TestCase):
    def test_non_root_suppresses_messages_by_default(self):
        stream = io.StringIO()
        logger = RunLogger(stream, rank=3, root_rank=0, verbosity="debug")

        logger.info("ordinary")
        logger.high("detail")
        logger.debug("debug")
        logger.warning("warning")
        logger.error("error")
        logger.header(
            program="slw_exchange.x",
            version="test",
            calculation="j",
            prefix="sample",
            outdir=".",
            savedir="sample.save",
            mpi_size=4,
        )
        logger.backend("native.exchange", execution="MPI")
        logger.progress("Exchange integration", 1, 4)
        logger.footer("slw_exchange.x")

        self.assertEqual(stream.getvalue(), "")

    def test_quiet_keeps_root_warning_but_suppresses_information(self):
        stream = io.StringIO()
        logger = RunLogger(stream, rank=0, verbosity="quiet")

        logger.info("ordinary")
        logger.high("detail")
        logger.debug("debug")
        logger.warning("important")

        self.assertEqual(stream.getvalue(), "     WARNING: important\n")

    def test_phase_context_records_cpu_and_wall_time(self):
        wall_values = iter((10.0, 12.5))
        cpu_values = iter((3.0, 3.75))
        timers = TimerBook(
            wall_clock=lambda: next(wall_values),
            cpu_clock=lambda: next(cpu_values),
        )

        with timers.phase("integrate") as sample:
            self.assertEqual(sample.name, "integrate")

        record = timers.get("integrate")
        self.assertIsNotNone(record)
        self.assertEqual(record.calls, 1)
        self.assertAlmostEqual(record.cpu_seconds, 0.75)
        self.assertAlmostEqual(record.wall_seconds, 2.5)
        self.assertAlmostEqual(sample.cpu_seconds, 0.75)
        self.assertAlmostEqual(sample.wall_seconds, 2.5)

    def test_progress_is_line_based_and_honors_throttles(self):
        clock_values = iter((0.0, 0.1, 0.2, 0.3))
        stream = io.StringIO()
        logger = RunLogger(
            stream,
            progress_clock=lambda: next(clock_values),
        )

        self.assertTrue(logger.progress("Exchange integration", 1, 10, every=2))
        self.assertFalse(logger.progress("Exchange integration", 2, 10, every=2))
        self.assertTrue(logger.progress("Exchange integration", 3, 10, every=2))
        self.assertTrue(
            logger.progress(
                "Exchange integration",
                10,
                10,
                every=20,
                min_interval=60.0,
            )
        )

        self.assertEqual(
            stream.getvalue().splitlines(),
            [
                "     Exchange integration: completed 1/10 (10.0%)",
                "     Exchange integration: completed 3/10 (30.0%)",
                "     Exchange integration: completed 10/10 (100.0%)",
            ],
        )
        self.assertNotIn("\r", stream.getvalue())
        self.assertNotIn("\x1b", stream.getvalue())

    def test_backend_and_timed_footer_format(self):
        self.assertEqual(
            format_backend_line(
                "slw.exchange.native",
                execution="MPI",
                input_format="epr",
            ),
            "     backend     = slw.exchange.native [MPI, input_format=epr]",
        )

        timers = TimerBook()
        timers.record("total", cpu_seconds=1.5, wall_seconds=2.25)
        stream = io.StringIO()
        print_footer(stream, program="slw_exchange.x", timers=timers)
        output = stream.getvalue()
        self.assertIn("SLW_EXCHANGE : CPU 1.50s WALL 2.25s", output)
        self.assertIn("slw_exchange.x: JOB DONE.", output)


if __name__ == "__main__":
    unittest.main()
