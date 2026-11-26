"""Controls from the matched R36 search confirmation."""

NOTES = {
    "contract_check": "Reminds the executor to verify task requirements before declaring completion.",
    "observation_dict": "Stores repeated tool observations once per memory-consolidation request.",
}
CAPS = dict(execution=8192, adaptation=8192, retrieval=8192, task_extraction=4096,
            task_consolidation=4096, adaptation_update=4096)
TOKENIZER_HASHES = {'chat': '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3', 'embedding': '223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7'}
