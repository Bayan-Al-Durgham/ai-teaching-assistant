from pydantic_ai import Agent

from teaching_assistant import memory


def test_save_load_and_clear_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("TEACHING_ASSISTANT_DATA_DIR", str(tmp_path))

    probe_agent = Agent("test", name="memory_probe")
    result = probe_agent.run_sync("hello")

    assert memory.load_history("conv-1") == []

    memory.save_history("conv-1", result.all_messages())
    loaded = memory.load_history("conv-1")
    assert len(loaded) == len(result.all_messages())

    memory.clear_history("conv-1")
    assert memory.load_history("conv-1") == []


def test_conversations_are_isolated_by_id(monkeypatch, tmp_path):
    monkeypatch.setenv("TEACHING_ASSISTANT_DATA_DIR", str(tmp_path))

    probe_agent = Agent("test", name="memory_probe")
    result = probe_agent.run_sync("hello")

    memory.save_history("student-a", result.all_messages())
    assert memory.load_history("student-b") == []
    assert len(memory.load_history("student-a")) == len(result.all_messages())
