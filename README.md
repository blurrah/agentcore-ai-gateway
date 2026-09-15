# AgentCore + Vercel AI Gateway (Python)

An Amazon Bedrock AgentCore agent whose model calls go through Vercel AI Gateway
instead of straight to Bedrock. One credential for 100+ models, server-side
failover across models and providers, per-session usage attribution.

```
invoke.py ──POST /invocations──▶ agent.py (BedrockAgentCoreApp on :8080)
                                    │  Strands Agent + current_time tool
                                    ▼
                                 OpenAIModel(base_url="https://ai-gateway.vercel.sh/v1")
                                    │  Authorization: Bearer $AI_GATEWAY_API_KEY
                                    ▼
                              Vercel AI Gateway ──▶ Anthropic / Bedrock / Vertex / OpenAI / ...
```

## Files

| File | Shows |
| --- | --- |
| `agent.py` | The core example: AgentCore entrypoint, Strands agent, OpenAI-compatible client pointed at the Gateway, fallback models, session attribution |
| `invoke.py` | Calling the agent locally with the AgentCore session header |
| `examples/01_model_fallbacks.py` | `models`: try the next model when the primary fails |
| `examples/02_provider_routing.py` | `order`, `only`, `sort`: which upstream serves a given model |
| `examples/03_timeouts.py` | Gateway-side `providerTimeouts` plus client-side `httpx.Timeout` |
| `examples/04_agent_fast_failover.py` | `agent.py` with all of the above combined |
| `examples/05_agent_ai_sdk_python.py` | Same agent on the [AI SDK for Python](https://github.com/vercel-labs/ai-python) instead of Strands: Gateway by default, typed routing params |

`examples/01`–`03` use the plain `openai` SDK so the request body is visible with
no framework in the way. Everything under `providerOptions.gateway` is a Gateway
extension; the OpenAI SDK passes it through via `extra_body`.

### Strands or the AI SDK for Python?

AgentCore does not care; it only needs the `/invocations` and `/ping` contract.
Strands (`agent.py`) is the AWS-native choice and reaches the Gateway through
its OpenAI-compatible model class. The AI SDK for Python (`examples/05`) is
Vercel's own client: no `base_url`, `AI_GATEWAY_API_KEY` is read automatically,
and `models`/`order`/`only`/`sort`/`user` are typed fields on
`ai.RoutingParams` / `ai.InferenceRequestParams` instead of a raw dict.
`providerTimeouts` is not typed yet and still goes through `extra_body`. It is
in public beta.

## The one idea

The Gateway speaks the OpenAI Chat Completions API. Any OpenAI-compatible
client becomes a Gateway client by changing two arguments:

```python
OpenAIModel(
    client_args={
        "api_key": os.environ["AI_GATEWAY_API_KEY"],
        "base_url": "https://ai-gateway.vercel.sh/v1",
    },
    model_id="anthropic/claude-sonnet-4.6",
)
```

Model ids are `<creator>/<model>`. The agent code is identical for
`openai/gpt-5.4` or `google/gemini-3-flash`.

## Fallbacks, routing, timeouts

All of these live in `providerOptions.gateway` in the request body:

| Option | Level | Meaning |
| --- | --- | --- |
| `models: [...]` | model | If the primary fails on every provider, try these in order. `completion.model` says who answered. |
| `order: [...]` | provider | Preferred upstreams for each model. Unlisted ones remain a last resort. |
| `only: [...]` | provider | Hard allow-list, e.g. `["bedrock"]` to stay inside AWS. |
| `sort: "cost" \| "ttft" \| "tps"` | provider | Let the Gateway rank providers by price, first-token latency, or throughput. |
| `providerTimeouts.byok.<provider>: ms` | provider | Abandon a provider that has not started responding in time (1 000–789 000 ms; streaming stops the timer at the first chunk). BYOK keys only. |
| `user`, `tags` | reporting | Attribution in the Gateway dashboard. |

Client-side `timeout` and `max_retries` on the OpenAI client are a separate
layer that guards against the Gateway or network itself being slow. Setting
`max_retries=0` keeps the SDK from retrying on its own so the Gateway is the
only thing doing failover.

## Running it anyway

```bash
cp .env.example .env            # AI_GATEWAY_API_KEY from the Vercel dashboard
uv sync --group dev
set -a; source .env; set +a
uv run agent.py                 # terminal 1
uv run invoke.py "What time is it in Amsterdam?"   # terminal 2
uv run pytest                   # offline; asserts the exact request sent to the Gateway
```

## References

- [Python with AI Gateway](https://vercel.com/docs/ai-gateway/sdks-and-apis/python)
- [Model fallbacks](https://vercel.com/docs/ai-gateway/models-and-providers/model-fallbacks)
- [Provider routing: order, only, sort](https://vercel.com/docs/ai-gateway/models-and-providers/provider-options)
- [Provider timeouts](https://vercel.com/docs/ai-gateway/models-and-providers/provider-timeouts)
- [bedrock-agentcore-sdk-python](https://github.com/aws/bedrock-agentcore-sdk-python)
