"""Run an offline evaluation over one or more JSON datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clients.mock_agent_client import MockAgentClient
from evaluators import EvaluationEngine
from models.api_models import AgentRequest
from utils.data_loader import load_json_cases
from utils.reporting import write_json_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", help="JSON dataset files to execute")
    parser.add_argument("--report", default="reports/evaluation_report.json")
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Write the report but return exit code 0 even when quality gates fail.",
    )
    args = parser.parse_args()

    client = MockAgentClient()
    engine = EvaluationEngine()
    reports = []

    for dataset_path in args.datasets:
        for case in load_json_cases(dataset_path):
            response = client.generate(
                AgentRequest(prompt=case.input, metadata=case.metadata),
            )
            case.actual_output = response.output
            case.retrieved_context = response.retrieved_context
            reports.append(engine.evaluate(case))

    write_json_report(reports, args.report)
    failed = [report for report in reports if not report.passed]
    print(f"Wrote {len(reports)} evaluations to {Path(args.report).resolve()}")
    if failed:
        print(f"{len(failed)} report(s) failed quality gates.")
    return 0 if args.allow_failures or not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
