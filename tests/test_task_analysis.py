from llmperf.task_analysis import build_task_analyses


def test_generic_task_join():
    instances = [
        {
            "task_instance_id": "instance-1",
            "task_definition_id": "definition-1",
            "state": "completed",
            "spec": {"dimensions": {"delay": 60}, "trial_index": 0},
        }
    ]
    dispatches = [
        {
            "task_instance_id": "instance-1",
            "node_id": "warm",
            "runner_id": "runner-1",
            "lineage": {"role": "warm", "payload_id": "replay"},
        }
    ]
    runners = {"runner-1": {"status": "succeeded", "summary": {"results": {}}}}

    analyses = build_task_analyses(instances, dispatches, runners)

    assert analyses[0]["spec"]["dimensions"] == {"delay": 60}
    assert analyses[0]["nodes"][0]["node_id"] == "warm"
    assert analyses[0]["nodes"][0]["runner"]["status"] == "succeeded"
