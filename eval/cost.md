# Paid calls: estimate before «плати» (2026-10-20, AWS public price list us-east-1, on-demand)

Prices read 2026-10-20 23:1xZ from `pricing.us-east-1.amazonaws.com/offers/v1.0/aws/{AmazonBedrock,transcribe,AmazonPolly}/current/us-east-1/index.json`:

| Item | Price |
|---|---|
| Nova Micro | in $0.000035 / 1K tok · out $0.00014 / 1K |
| Nova Lite | in $0.00006 / 1K · out $0.00024 / 1K |
| Nova Pro | in $0.0008 / 1K · out $0.0032 / 1K |
| Transcribe streaming | $0.0001667 / s (`USE1-StreamingAudio`); estimate below uses $0.0004 / s (= $0.024/min) and a 15 s minimum per request, worst case |
| Polly neural | $16 / 1M chars |

| Use | Volume | Estimate |
|---|---|---|
| Transcribe: 120 recorded phrases, one pass, cached | 120 × 15 s worst case | ≤ $0.72 |
| Transcribe: live demo and video takes | ~30 min | ≤ $0.72 |
| Bedrock per eval run: router (Micro/Lite/Pro) + baseline a (Pro on all, 49 frames) + baseline b (Micro) | ~2,400 tok per frame call, ~400 per text call, ~60 out | ≈ $0.25 |
| Bedrock: ~4 runs (tune, held-out, repeats) ×2 margin for image tokens | | ≈ $2 |
| Polly neural: ~120 replies × 80 chars × 3 passes | ~30K chars | ≈ $0.50 |
| **Ceiling to ask for** | | **$5** |

Nova first-party models are billed to AWS credits; Claude on Bedrock comes through AWS Marketplace and is not covered, so this build uses Nova only. Bedrock cross-Region inference is not covered by Free Tier credits, so Nova is called with in-Region model IDs, not a `us.` profile.

## Budget and the cap

- Budget for the whole build: $5 of AWS credits.
- Coverage, AWS page aws.amazon.com/free/machine-learning/ (read 2026-10-21): Bedrock "Use credits to access Amazon Bedrock features in both Free and Paid plans"; Polly "Use credits to access features in the Free and Paid plan"; Transcribe "12-month free trial with the Paid plan… 60 minutes per month".
- Ceiling in code: `src/n8/spend.py` (`SpendCap.charge` before every paid call, ledger `eval/out/spend.jsonl`, cumulative, $5).
