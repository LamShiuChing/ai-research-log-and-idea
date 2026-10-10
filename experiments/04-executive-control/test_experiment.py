"""Non-model checks of information isolation, task semantics and run constraints."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import experiment as e


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.data = e.datasets()

    def test_reference_answers_and_retrieval_coverage(self):
        for suites in self.data.values():
            for data in suites.values():
                store = e.PassiveStore(data["corpus"])
                for task in data["tasks"]:
                    obs = task["observation"]
                    selected = store.search(obs["entity"])
                    exact = [r for r in data["corpus"] if r["entity"] == obs["entity"]]
                    self.assertEqual(len(selected), len(exact))
                    # Labels constructed by scenario, rather than by this resolver.
                    result = e.resolve(obs, obs["current"]+selected)
                    self.assertEqual(result["answer"], task["expected"])
                    self.assertTrue(e.score(task, result, obs["current"]+selected)["evidence_valid"])
                    local = e.resolve(obs, obs["current"])
                    self.assertEqual(local["answer"] != "ABSTAIN", task["current_sufficient"])
                    if task["needs_memory"]:
                        self.assertNotEqual(local["answer"], task["expected"])

    def test_splits_disjoint_and_observation_allowlist(self):
        entities, ids, codes = set(), set(), set()
        for suites in self.data.values():
            for data in suites.values():
                local_entities = {t["observation"]["entity"] for t in data["tasks"]}
                records = data["corpus"] + [r for t in data["tasks"] for r in t["observation"]["current"]]
                local_ids, local_codes = {r["id"] for r in records}, {r["value"] for r in records}
                for previous, current in ((entities,local_entities),(ids,local_ids),(codes,local_codes)):
                    self.assertFalse(previous & current)
                    previous.update(current)
                for task in data["tasks"]:
                    self.assertEqual(set(task["observation"]), {"entity", "fields", "as_of", "current"})

    def test_passive_store_immutability_and_no_filtering(self):
        data = self.data["development"]["access"]
        source = copy.deepcopy(data["corpus"])
        store = e.PassiveStore(source)
        before = e.digest(store.read_all())
        source.clear()
        store.read_all()[0]["value"] = "tampered"
        for task in data["tasks"]:
            result = store.search(task["observation"]["entity"])
            if result:
                result[0]["source"] = "tampered"
        self.assertEqual(before, e.digest(store.read_all()))
        self.assertTrue(any(r["source"] == "chat" for r in store.read_all()))
        self.assertTrue(any(r["expires"] < "2026-09-10" for r in store.read_all()))

    def test_full_context_preflight_all_splits(self):
        for suites in self.data.values():
            for data in suites.values():
                for task in data["tasks"]:
                    text = e.ANSWER_SYSTEM+"MEMORY RECORDS\n"+e.packed(data["corpus"])+"\nCURRENT TASK\n"+e.packed(task["observation"])
                    self.assertLessEqual(len(text.encode())+256+96, e.CONFIG["num_ctx"])

    def test_future_unofficial_conflict_and_bad_citations(self):
        task = next(t for t in self.data["development"]["access"]["tasks"] if t["scenario"] == "current")
        obs = task["observation"]
        good = obs["current"][0]
        for change in ({"issued":"2027-01-01"},{"source":"chat"},{"expires":"2025-01-01"},{"entity":"wrong"}):
            record = dict(good, **change)
            self.assertEqual(e.resolve(obs,[record])["answer"],"ABSTAIN")
        conflict = dict(good, value="OTHER", id="conflict")
        self.assertEqual(e.resolve(obs,[good,conflict])["answer"],"ABSTAIN")
        self.assertFalse(e.score(task, dict(answer=task["expected"],evidence=["invented"]),[good])["evidence_valid"])

    def test_invalid_gate_does_not_fallback_or_search(self):
        task = self.data["development"]["access"]["tasks"][0]
        with tempfile.TemporaryDirectory() as tmp:
            runner = e.Runner(Path(tmp))
            store = e.PassiveStore([])
            with patch.object(runner,"generate",side_effect=ValueError("bad gate")), patch.object(store,"search") as search:
                row = runner.trial(task,store,"model_gate","development")
                search.assert_not_called()
                self.assertFalse(row["valid"])
                self.assertFalse(row["correct"])

    def test_parsers_and_budget(self):
        for raw in ('{"decision":"search"}', '{"decision":"act","expected":1}', '[]'):
            with self.assertRaises(ValueError):
                e.parse(raw, True)
        with tempfile.TemporaryDirectory() as tmp:
            runner = e.Runner(Path(tmp))
            runner.calls = e.CONFIG["max_calls"]
            with self.assertRaises(RuntimeError):
                runner.check_budget()


if __name__ == "__main__":
    unittest.main()
