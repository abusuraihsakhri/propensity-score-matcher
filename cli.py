#!/usr/bin/env python3
"""
CLI for Propensity Score Matcher.

Usage:
  python cli.py match --treatment 1 1 0 0 1 0 --covariates '[[0.5,65],[1.2,70],[0.8,55],[1.5,60],[0.3,72],[1.1,68]]'
  python cli.py balance --treatment 1 1 0 0 1 0 --covariates '[[0.5],[1.2],[0.8],[1.5],[0.3],[1.1]]'
  python cli.py batch --input sample.csv --output results.csv
  python cli.py ps --treatment 1 1 0 0 --covariates '[[0.5],[1.2],[0.8],[1.5]]'
  python cli.py audit --task-id TASK-001
  python cli.py chat "Explain the methodology"
  python cli.py verify-audit
"""

import argparse
import json
import sys

from ps_matcher import (
    propensity_score_match,
    estimate_propensity_scores,
    covariate_balance,
    summary,
    process_csv,
)

from agents.supervisor import SystemSupervisor
from agents.base import AuditLogger
from agents.models import SystemTaskPayload


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ps-matcher",
        description="Propensity Score Matcher",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- match ---
    p_match = sub.add_parser("match", help="Run propensity score matching")
    p_match.add_argument("--treatment", type=int, nargs="+", required=True, help="Treatment indicators (1/0)")
    p_match.add_argument("--covariates", type=str, required=True, help="JSON array of covariate arrays")
    p_match.add_argument("--outcome", type=float, nargs="+", help="Outcome values (optional)")
    p_match.add_argument("--method", choices=["nearest", "optimal", "stratification"], default="nearest")
    p_match.add_argument("--caliper", type=float, default=0.2, help="Caliper for matching")
    p_match.add_argument("--replace", action="store_true", help="Match with replacement")
    p_match.add_argument("--labels", type=str, nargs="+", help="Covariate labels")
    p_match.add_argument("--json", action="store_true", help="Output as JSON")

    # --- balance ---
    p_bal = sub.add_parser("balance", help="Check covariate balance")
    p_bal.add_argument("--treatment", type=int, nargs="+", required=True)
    p_bal.add_argument("--covariates", type=str, required=True)
    p_bal.add_argument("--labels", type=str, nargs="+")

    # --- ps ---
    p_ps = sub.add_parser("ps", help="Estimate propensity scores")
    p_ps.add_argument("--treatment", type=int, nargs="+", required=True)
    p_ps.add_argument("--covariates", type=str, required=True)

    # --- batch ---
    p_batch = sub.add_parser("batch", help="Process CSV file")
    p_batch.add_argument("-i", "--input", required=True)
    p_batch.add_argument("-o", "--output", default="results.csv")
    p_batch.add_argument("--method", choices=["nearest", "optimal", "stratification"], default="nearest")
    p_batch.add_argument("--caliper", type=float, default=0.2)

    # --- audit ---
    p_audit = sub.add_parser("audit", help="Run audit task through supervisor")
    p_audit.add_argument("--task-id", required=True, help="Task identifier")
    p_audit.add_argument("--target", default="GENERIC-TARGET", help="Target identifier")
    p_audit.add_argument("--primary", type=float, default=10.0, help="Primary metric")
    p_audit.add_argument("--secondary", type=float, default=5.0, help="Secondary metric")

    # --- chat ---
    p_chat = sub.add_parser("chat", help="Supervisory LLM chat")
    p_chat.add_argument("query", nargs="+", help="Query text for the supervisor")

    # --- verify-audit ---
    sub.add_parser("verify-audit", help="Verify HMAC audit trail integrity")

    args = parser.parse_args(argv)

    if args.command == "match":
        covariates = json.loads(args.covariates)
        result = propensity_score_match(
            args.treatment, covariates, args.outcome,
            method=args.method, caliper=args.caliper,
            replace=args.replace, labels=args.labels
        )
        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            print(summary(args.treatment, covariates, args.outcome, labels=args.labels))
        return 0

    elif args.command == "balance":
        covariates = json.loads(args.covariates)
        result = covariate_balance(args.treatment, covariates, labels=args.labels)
        print("Covariate Balance Assessment")
        print("=" * 50)
        for b in result["balance"]:
            status = "OK" if b["balanced_before"] else "IMBALANCED"
            print(f"  {b['covariate']:<15} SMD={b['smd_before']:.4f} [{status}]")
        return 0

    elif args.command == "ps":
        covariates = json.loads(args.covariates)
        ps = estimate_propensity_scores(args.treatment, covariates)
        for i, p in enumerate(ps):
            print(f"  Subject {i}: PS={p:.6f} (treatment={args.treatment[i]})")
        return 0

    elif args.command == "batch":
        result = process_csv(args.input, args.output, method=args.method, caliper=args.caliper)
        print(f"Processed {result['n_treated']} treated + {result['n_control']} control -> {args.output}")
        return 0

    elif args.command == "audit":
        supervisor = SystemSupervisor(model_provider="mock")
        payload = SystemTaskPayload(
            task_id=args.task_id,
            target_identifier=args.target,
            primary_metric=args.primary,
            secondary_metric=args.secondary,
        )
        dossier = supervisor.process_task(payload)
        print(json.dumps(dossier.to_dict(), indent=2, default=str))
        return 0

    elif args.command == "chat":
        supervisor = SystemSupervisor(model_provider="mock")
        query = " ".join(args.query)
        response = supervisor.query_supervisory_chat(query)
        print(response)
        return 0

    elif args.command == "verify-audit":
        valid = AuditLogger.verify_integrity()
        print(f"Audit trail integrity: {'VALID' if valid else 'INVALID'}")
        print(f"Audit blocks: {len(AuditLogger.get_trail())}")
        return 0 if valid else 1

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
