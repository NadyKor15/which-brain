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

Account 762071317428 (default until told otherwise); Nova first-party is inside AWS credits (brief §3: $189.84, of which $36.32 expire 31.10). Claude via Marketplace is excluded. Billing card change is Mikhail's (rule 55). Default AWS CLI profile on Studio returns InvalidClientTokenId; the working key is Keychain `aws-fleetmemory-cli`.

## Approved and credit check (2026-10-21)
- Mikhail «Да» 2026-10-21T10:49:54Z: N8 up to **$5, AWS credits only** (docs/DECISIONS.md in the campaign repo).
- Check before 2026-10-21T10:52:36Z (written then), free read-only APIs, IAM fleetmemory-cli: `freetier get-account-plan-state` → plan **PAID**, ACTIVE, Free Tier credits **$153.52** (= $189.84 − $36.32: the API shows only Free Tier credits, not the grant that expires 31.10). Paid plan → anything above credits goes to the card, so the ceiling matters.
- Coverage, AWS page aws.amazon.com/free/machine-learning/ (read 21.10): Bedrock «Use credits to access Amazon Bedrock features in both Free and Paid plans»; Polly «Use credits to access features in the Free and Paid plan»; Transcribe «12-month free trial with the Paid plan… 60 minutes per month». Claude via Marketplace stays out → Nova only. Bedrock cross-Region inference is not supported on Free Tier credits → call Nova with the in-Region model ID, not a `us.` profile.
- Ceiling in code: `src/n8/spend.py` (`SpendCap.charge` before every paid call, ledger `eval/out/spend.jsonl`, cumulative, $5). Spend after each run → one line to the orchestrator.
