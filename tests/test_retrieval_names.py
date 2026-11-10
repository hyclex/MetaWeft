"""Optional task names preserve retrieval metadata without influencing ranking."""

import unittest
from unittest.mock import Mock

from MetaWeft.src.memory_retrieval import load_preset, retrieve
from MetaWeft.src.memory_retrieval.candidate_search import prepare_exact_bank
from MetaWeft.tests.test_memory_retrieval_candidate_search import entry
from MetaWeft.tests.test_memory_retrieval_v2 import Bank, Ranker


class RetrievalNameTests(unittest.TestCase):
    def test_exact_probe_name_changes_and_presence_mismatches_fail_before_model_calls(self):
        for probe_name, export_name in (("Before", "After"), ("Before", None), (None, "After")):
            with self.subTest(probe=probe_name, exported=export_name):
                probe = entry("a")["document"]
                exported = entry("a")
                if probe_name is not None:
                    probe["name"] = probe_name
                if export_name is not None:
                    exported["document"]["name"] = export_name
                store = Mock()
                store.query.return_value = {"hits": {"hits": [{"_id": "a", "_source": probe}]}}
                store.export.return_value = [exported]
                model = Mock(dimensions=2, max_retries=5, retry_delay_seconds=60)
                model.embed.side_effect = AssertionError("A contradictory bank must fail before embedding")
                with self.assertRaisesRegex(ValueError, "probe"):
                    retrieve("task", 10, store=store, llm=model, meta=load_preset("exact_base"), record_trace=lambda _: None)
                model.embed.assert_not_called()
                model.chat.assert_not_called()

    def test_invalid_optional_names_are_rejected_in_both_backends(self):
        for name in (None, 4, "", "  ", []):
            with self.subTest(name=name):
                bank, model = Bank(1), Ranker()
                model.max_retries = 5
                bank.hits[0]["_source"]["name"] = name
                result = retrieve("task", 10, store=bank, llm=model, meta=load_preset("efficiency_1"), record_trace=lambda _: None)
                self.assertEqual(result.status, "degraded")
                self.assertEqual(result.records, [])
                self.assertEqual(model.chats, [])
                row = entry("a")
                row["document"]["name"] = name
                with self.assertRaisesRegex(ValueError, "name"):
                    prepare_exact_bank(Mock(export=Mock(return_value=[row])),
                                       representations=["combined"], dimensions=2)


if __name__ == "__main__":
    unittest.main()
