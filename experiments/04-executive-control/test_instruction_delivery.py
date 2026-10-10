import json
import unittest

import experiment as v1
import instruction_delivery as d


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.data = d.make_data()

    def test_reference_and_search_coverage(self):
        for suites in self.data["phases"].values():
            for dataset in suites.values():
                store = v1.PassiveStore(dataset["corpus"])
                for task in dataset["tasks"]:
                    obs = task["observation"]
                    memory = store.search(obs["entity"])
                    self.assertEqual(len(memory),sum(r["entity"] == obs["entity"] for r in dataset["corpus"]))
                    self.assertEqual(v1.resolve(obs,obs["current"]+memory)["answer"],task["expected"])

    def test_split_separation(self):
        previous = [set(),set(),set()]
        for suites in self.data["phases"].values():
            for dataset in suites.values():
                records = dataset["corpus"]+[r for t in dataset["tasks"] for r in t["observation"]["current"]]
                sets = [{t["observation"]["entity"] for t in dataset["tasks"]}, {r["id"] for r in records},{r["value"] for r in records}]
                for seen,current in zip(previous,sets):
                    self.assertFalse(seen & current)
                    seen.update(current)
                self.assertTrue(all(r["id"].startswith("v2r-") and r["value"].startswith("C") for r in records))

    def test_nonce_negative_no_marker_leak(self):
        for task in self.data["controls"]:
            request = d.request_for("chat_none",task["instruction"],d.NONCE_PAYLOAD,d.NONCE_SCHEMA)
            self.assertNotIn(task["marker"],json.dumps(request))
            for route in d.POSITIVE:
                self.assertIn(task["marker"],json.dumps(d.request_for(route,task["instruction"],d.NONCE_PAYLOAD,d.NONCE_SCHEMA)))
            self.assertNotIn(task["marker"],json.dumps(d.NONCE_SCHEMA))

    def test_exact_delivery_equivalence_and_raw_boundary(self):
        text,payload = "Instruction abc", "Task def"
        system = d.request_for("chat_system",text,payload,d.NONCE_SCHEMA)["request"]
        user = d.request_for("chat_user",text,payload,d.NONCE_SCHEMA)["request"]
        raw = d.request_for("raw_system",text,payload,d.NONCE_SCHEMA)["request"]
        self.assertEqual(system["messages"],[dict(role="system",content=text),dict(role="user",content=payload)])
        self.assertEqual(user["messages"],[dict(role="user",content=text+"\n\n"+payload)])
        self.assertEqual(raw["prompt"],"<|im_start|>system\nInstruction abc<|im_end|>\n<|im_start|>user\nTask def<|im_end|>\n<|im_start|>assistant\n")
        self.assertTrue(raw["raw"])
        self.assertEqual(system["format"],user["format"])
        self.assertEqual(system["options"],raw["options"])

    def test_plan_context_and_hidden_labels(self):
        plans = d.plan_requests(self.data)
        self.assertEqual(len(plans),152)
        for key,plan in plans.items():
            if key.startswith("controls"):
                continue
            # Schema and record values are legitimate observations, labels are not.
            request = plan["request"]
            body = request.get("prompt", "") + "\n".join(m["content"] for m in request.get("messages",[]))
            for label in ('"scenario"','"expected"','"suite"','"needs_memory"'):
                self.assertNotIn(label,body)
        with self.assertRaises(ValueError):
            d.request_for("raw_system","a"*17000,"",d.NONCE_SCHEMA)


if __name__ == "__main__":
    unittest.main()
