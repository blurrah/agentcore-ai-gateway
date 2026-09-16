"""Attribute Gateway spend to an application user and project, then query it.

The report endpoint is available on Pro and Enterprise plans and can lag writes by
a few minutes. Run once to create usage, then rerun with --report-only.
"""

import argparse
import os
from datetime import UTC, datetime

import httpx
from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="customer-42", help="Your application's user ID")
    parser.add_argument("--project", default="support-agent", help="Your application's project ID")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


args = parse_args()
api_key = os.environ["AI_GATEWAY_API_KEY"]
project_tag = f"project:{args.project}"

if not args.report_only:
    client = OpenAI(api_key=api_key, base_url="https://ai-gateway.vercel.sh/v1")
    completion = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Reply with: usage recorded"}],
        extra_body={
            "providerOptions": {
                "gateway": {
                    "user": args.user,
                    "tags": [project_tag, "feature:chat", "app:agentcore-ai-gateway"],
                }
            }
        },
    )
    print(completion.choices[0].message.content)
    print(f"attributed to user={args.user!r}, project={args.project!r}")

# Reports are account-scoped. Filter by the project tag, then group by app user.
today = datetime.now(UTC).date().isoformat()
response = httpx.get(
    "https://ai-gateway.vercel.sh/v1/report",
    headers={"Authorization": f"Bearer {api_key}"},
    params={
        "start_date": today,
        "end_date": today,
        "group_by": "user",
        "tags": project_tag,
        "tags_match": "all",
        "api_key_id": "self",
    },
    timeout=20,
)
response.raise_for_status()
rows = response.json()["results"]

if not rows:
    print("No report rows yet. Reporting can lag by a few minutes; rerun with --report-only.")
else:
    print(f"\nUsage for {project_tag} on {today}:")
    for row in rows:
        print(
            f"{row['user']}: {row['request_count']} requests, "
            f"{row['input_tokens']} input tokens, "
            f"{row['output_tokens']} output tokens, "
            f"${row['total_cost']:.6f}"
        )
