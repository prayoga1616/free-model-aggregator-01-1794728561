# free-model-aggregator-01 — one deterministic aggregator server

Entry for the make.fun bounty **"The best free AI model aggregator"** by **@hanjp**.

## What it does

`main.ts` (Node 22, stdlib only, no dependencies) is a single server that:

1. **Reads the brief** (`$MAKEFUN_PROMPT_FILE`): the provider directory, the
   quality table (per-request best free model + rating), and the workload
   manifest with each request's requirements (tools / vision / JSON mode /
   minimum context) and weights.
2. **Probes every provider's catalog route** (`/v1/models`, gemini
   `/v1beta/models`), speaking each provider's auth style (bearer, `x-api-key`,
   `x-goog-api-key`).
3. **Registers one catalog** with the environment: exactly the free models the
   providers actually list, in provider order. `node main.ts --list` prints the
   same document.
4. **Serves every client request** on `/v1/chat/completions` (OpenAI wire),
   `/v1/messages` (Anthropic wire) and `/v1/models` — streaming and
   non-streaming — by routing to the highest-rated free model that meets the
   request's requirements, translating across wires when the best model speaks
   a different protocol (OpenAI ↔ Anthropic ↔ Gemini), and failing over down
   the rating order when a provider refuses. Providers whose free-tier budget
   is spent are retired for the rest of the trial; temporary refusals are
   retried on later requests.

## Design notes

- **Deterministic**: no wall clock, no randomness. Every choice is a function
  of the brief, the requests and the providers' answers. Timestamps in client
  answers are fixed.
- **Failover is budget-aware**: a refusal that says the budget is spent marks
  the provider permanently exhausted; `concurrent_requests` and one-off
  refusals do not.
- **Weight-aware routing**: requests arrive in manifest order; the manifest's
  weights ride along in the state export the scorer reads.
- **Cross-wire translation** carries tools, vision (base64 and URL images),
  JSON mode, stop sequences and max tokens through all six direction pairs.

## Local verification

Graded with the bounty kit's practice trials (env + scorer), 20/20 valid
requests served on every trial:

```
trial-0: score 99 (quality 99, served 100, wire 100, failover 100, catalog 100)
trial-1: score 99 (quality 99, served 100, wire 100, failover 100, catalog 100)
trial-2: score 99 (quality 99, served 100, wire 100, failover 100, catalog 100)
```

The last quality point is hidden-budget luck: the top-rated free model is
oversubscribed across the workload, so some weight-1 requests necessarily fall
to the next provider once its budget is spent.

## License

MIT — see [LICENSE](LICENSE).
