import os
import tempfile
import unittest

from slw.cli.namelist import NamelistError, expand_placeholders, loads
from slw.cli.schema import parse_run_config


class NamelistTests(unittest.TestCase):
    def test_qe_types_and_comments(self):
        groups = loads(
            """
            &control
              calculation = 'j_tensor', ! inline comment
              prefix = 'sample'
            /
            &exchange
              input_format = 'epr',
              kmesh = 4, 4, 2,
              rotate = .false.
            /
            """
        )
        self.assertEqual(groups["control"]["calculation"], "j_tensor")
        self.assertEqual(groups["exchange"]["kmesh"], [4, 4, 2])
        self.assertIs(groups["exchange"]["rotate"], False)

    def test_schema_expands_only_explicit_workflow_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            groups = loads(
                """
                &control
                  calculation = 'gkq',
                  prefix = 'sample',
                  outdir = 'scratch'
                /
                &epr
                  epr = 'input.h5',
                  out = '${savedir}/gkq.h5'
                /
                """
            )
            config = parse_run_config(groups, stage="epr", cwd=directory)
            expected = os.path.join(directory, "scratch", "sample.save", "gkq.h5")
            self.assertEqual(config.parameters["out"], expected)

    def test_unknown_placeholder_is_rejected(self):
        with self.assertRaisesRegex(NamelistError, "unknown path placeholder"):
            expand_placeholders("${material}/file.h5", {"prefix": "sample"})

    def test_duplicate_stage_parameter_is_rejected(self):
        groups = loads(
            """
            &control calculation = 'gkq' /
            &input epr = 'a.h5' /
            &epr epr = 'b.h5' /
            """
        )
        with self.assertRaisesRegex(NamelistError, "defined in both"):
            parse_run_config(groups, stage="epr")


if __name__ == "__main__":
    unittest.main()
