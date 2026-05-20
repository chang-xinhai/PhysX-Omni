# Tiny Smoke Test

Run:

```bash
bash benchmark/scripts/run_tiny_smoke_test.sh
```

The script creates a tiny generated fixture under
`benchmark/tiny_example/generated/`, builds an RQS manifest, writes one fake VLM
`result.json`, runs aggregation, and validates denominator counts.

It does not require a GPU or a downloaded VLM. Use it to verify that the local
Python environment can run the manifest, aggregation, and validation layers
before launching large benchmark jobs.
