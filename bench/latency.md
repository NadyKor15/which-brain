# Time to first word and cost (20261022T145951Z)

Columns marked (from end of speech) run on one clock that starts when the cook stops talking in the recording: Transcribe final → Decider Request 1 (first sound = earcon) → Bedrock first token ∥ Decider Request 2 → answer ready → Polly first byte = first word. Columns marked (call) are single-call durations. Decider runs on a Mac mini over an SSH tunnel; mini hardware is not in $.

## router (e2e-router-text-20261022T145734Z.jsonl + e2e-router-frames-20261022T145810Z.jsonl + e2e-router-h-20261022T145815Z.jsonl, n=120)
| path | n | Transcribe final (from end of speech) | Decider R1 (call) | first sound (from end of speech) | Bedrock 1st token (from end of speech) | Decider R2 (call) | answer ready (from end of speech) | Polly 1st byte (call) | **first word (from end of speech)** p50 / p90 |
|---|---|---|---|---|---|---|---|---|---|
| all | 120 | 223.8 / 347.2 | 339.6 / 365.3 | 581.6 / 668.5 | 980.9 / 1524.7 | 469.3 / 519.3 | 635.0 / 1504.3 | 64.6 / 96.9 | **820.9 / 1587.8** |
| ask | 12 | 246.1 / 341.9 | 333.9 / 360.2 | 588.5 / 662.2 | – | – | 588.5 / 662.2 | 62.4 / 68.0 | **651.7 / 827.1** |
| image gate | 1 | 250.9 / 250.9 | 332.4 / 332.4 | 583.4 / 583.4 | – | – | 583.4 / 583.4 | 66.0 / 66.0 | **649.4 / 649.4** |
| look | 12 | 216.9 / 314.2 | 360.2 / 365.3 | 563.2 / 668.3 | 1438.0 / 1572.2 | 472.2 / 502.9 | 1510.6 / 1608.0 | 65.3 / 71.1 | **1575.3 / 1672.8** |
| quiet_answer | 24 | 243.3 / 277.1 | 347.1 / 357.2 | 583.6 / 633.2 | 917.5 / 1008.4 | 469.3 / 519.3 | 987.8 / 1201.5 | 67.0 / 108.8 | **1089.1 / 1263.5** |
| read_closely | 9 | 202.5 / 243.5 | 339.0 / 354.0 | 538.2 / 607.8 | 1460.7 / 1684.9 | 467.4 / 494.7 | 1627.4 / 1902.2 | 64.6 / 83.2 | **1688.3 / 1985.4** |
| refuse | 42 | 235.6 / 347.2 | 336.6 / 370.4 | 579.3 / 710.7 | 964.9 / 964.9 | – | 579.3 / 722.6 | 63.2 / 99.2 | **645.7 / 787.4** |
| silent | 20 | 166.6 / 571.1 | 318.9 / 318.9 | – | – | – | 166.8 / 571.2 | – | **–** |

$ per 1,000 phrases: Bedrock 0.1121 (from the usage each call returned) · Transcribe streaming at $0.0001667/s (AWS price list, eval/cost.md): per second 0.467, or 2.5005 if a 15 s minimum per request applies (not re-checked on the pricing page in this run) · Polly neural ($16/1M chars) 0.6255 · **full stack 1.2046 – 3.2381** (Lambda not built).

## pro (e2e-pro-text-20261022T133634Z.jsonl + e2e-pro-frames-20261022T145134Z.jsonl + e2e-pro-h-20261022T145820Z.jsonl, n=120)
| path | n | Transcribe final (from end of speech) | Decider R1 (call) | first sound (from end of speech) | Bedrock 1st token (from end of speech) | Decider R2 (call) | answer ready (from end of speech) | Polly 1st byte (call) | **first word (from end of speech)** p50 / p90 |
|---|---|---|---|---|---|---|---|---|---|
| all | 120 | 223.8 / 347.2 | – | – | 1006.7 / 3172.5 | – | 1153.1 / 3281.6 | 69.8 / 122.5 | **1227.5 / 3418.7** |
| pro | 120 | 223.8 / 347.2 | – | – | 1006.7 / 3172.5 | – | 1153.1 / 3281.6 | 69.8 / 122.5 | **1227.5 / 3418.7** |

$ per 1,000 phrases: Bedrock 0.832 (from the usage each call returned) · Transcribe streaming at $0.0001667/s (AWS price list, eval/cost.md): per second 0.467, or 2.5005 if a 15 s minimum per request applies (not re-checked on the pricing page in this run) · Polly neural ($16/1M chars) 0.9828 · **full stack 2.2818 – 4.3153** (Lambda not built).

## classifier (e2e-classifier-text-20261022T134549Z.jsonl + e2e-classifier-frames-20261022T145353Z.jsonl + e2e-classifier-h-20261022T145827Z.jsonl, n=120)
| path | n | Transcribe final (from end of speech) | Decider R1 (call) | first sound (from end of speech) | Bedrock 1st token (from end of speech) | Decider R2 (call) | answer ready (from end of speech) | Polly 1st byte (call) | **first word (from end of speech)** p50 / p90 |
|---|---|---|---|---|---|---|---|---|---|
| all | 120 | 223.8 / 347.2 | – | 791.0 / 1058.3 | 1792.9 / 2317.8 | – | 1217.0 / 2150.7 | 67.9 / 109.1 | **1548.3 / 2362.0** |
| error | 5 | 252.8 / 271.5 | – | 812.1 / 1004.6 | – | – | 1477.8 / 1697.7 | 67.0 / 79.1 | **1541.3 / 1763.6** |
| image gate | 2 | 196.6 / 250.9 | – | 801.1 / 832.3 | – | – | 836.7 / 922.0 | 65.3 / 67.5 | **904.2 / 987.3** |
| look | 24 | 226.9 / 314.2 | – | 807.5 / 937.5 | 1910.0 / 2393.5 | – | 1952.6 / 2424.6 | 65.7 / 113.0 | **2068.5 / 2558.9** |
| quiet_answer | 17 | 237.3 / 258.6 | – | 759.7 / 916.0 | 1358.8 / 1792.9 | – | 1534.4 / 1837.4 | 70.1 / 109.1 | **1604.0 / 1903.9** |
| read_closely | 6 | 187.5 / 221.2 | – | 700.4 / 993.6 | 1809.5 / 2200.1 | – | 2010.3 / 2339.8 | 65.8 / 74.9 | **2074.5 / 2414.7** |
| refuse | 20 | 264.0 / 347.2 | – | 809.9 / 1069.3 | – | – | 809.9 / 1069.4 | 70.7 / 103.1 | **878.3 / 1139.3** |
| silent | 46 | 222.7 / 391.9 | – | – | – | – | 813.6 / 1287.1 | – | **–** |

$ per 1,000 phrases: Bedrock 0.1075 (from the usage each call returned) · Transcribe streaming at $0.0001667/s (AWS price list, eval/cost.md): per second 0.467, or 2.5005 if a 15 s minimum per request applies (not re-checked on the pricing page in this run) · Polly neural ($16/1M chars) 0.4791 · **full stack 1.0536 – 3.0871** (Lambda not built).

Mac mini load when the bench ran: `11:00  up 9 days,  3:08, 1 user, load averages: 15.57 18.64 19.65` (shared with other jobs).
Tunnel row: 20 x GET /health over the SSH forward Studio -> Mac mini: p50 2.6 ms, p90 3.5 ms (not Lambda: the Lambda host was not built).