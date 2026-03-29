import json
import os
import subprocess
from pathlib import Path

BRIDGE_PATH = "/run/media/julian/ML3/claude-gemini-bridge/claude_gemini_bridge.py"
TEST_SESSION_ID = "test-session-1234-5678"
PROJECT_PATH = "/run/media/julian/ML2"

MOCK_EVENTS = [
    {
        "type": "user",
        "uuid": "u1",
        "timestamp": "2026-03-27T12:00:00.000Z",
        "message": {"role": "user", "content": [{"type": "text", "text": "Hello, can you read a file and create a task?"}]},
    },
    {
        "type": "assistant",
        "uuid": "a1",
        "timestamp": "2026-03-27T12:00:05.000Z",
        "message": {
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 100, "output_tokens": 50, "cache_read_input_tokens": 20},
            "content": [
                {"type": "thinking", "thinking": "I need to read the file and then create a task."},
                {"type": "text", "text": "Sure, let me do that."},
                {"type": "tool_use", "id": "call_read", "name": "Read",
                 "input": {"file_path": "test.txt"}},
                {"type": "tool_use", "id": "call_task", "name": "TaskCreate",
                 "input": {"subject": "Test Task", "description": "Do something", "activeForm": "Testing"}},
            ],
        },
    },
    {
        "type": "user",
        "uuid": "u2",
        "timestamp": "2026-03-27T12:00:10.000Z",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_read", "content": "File content here", "is_error": False},
        ]},
    },
    {
        "type": "user",
        "uuid": "u3",
        "timestamp": "2026-03-27T12:00:11.000Z",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_task", "content": "Task created with id=1", "is_error": False},
        ]},
    },
    {
        "type": "assistant",
        "uuid": "a2",
        "timestamp": "2026-03-27T12:00:15.000Z",
        "message": {
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 200, "output_tokens": 30, "cache_read_input_tokens": 100},
            "content": [
                {"type": "tool_use", "id": "call_update", "name": "TaskUpdate",
                 "input": {"taskId": "1", "status": "completed"}},
            ],
        },
    },
    {
        "type": "user",
        "uuid": "u4",
        "timestamp": "2026-03-27T12:00:20.000Z",
        "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_update", "content": "", "is_error": False},
        ]},
    },
    # Metadata entries that should be skipped
    {"type": "file-history-snapshot", "timestamp": "2026-03-27T12:00:21.000Z"},
    {"type": "system", "timestamp": "2026-03-27T12:00:22.000Z"},
]


def create_mock_claude_session():
    project_slug = PROJECT_PATH.strip("/").replace("/", "-")
    mock_dir = Path(os.path.expanduser("~")) / ".claude" / "projects" / f"-{project_slug}"
    mock_dir.mkdir(parents=True, exist_ok=True)
    mock_file = mock_dir / f"{TEST_SESSION_ID}.jsonl"
    with open(mock_file, "w") as f:
        for e in MOCK_EVENTS:
            f.write(json.dumps(e) + "\n")
    return mock_file


def run_test():
    print("--- Starting Bridge Parity Test ---")
    mock_file = create_mock_claude_session()
    print(f"Created mock Claude session: {mock_file}")

    # Claude -> Gemini
    subprocess.run(["python3", BRIDGE_PATH, "-i", "claude", "-e", "gemini",
                    "--id", TEST_SESSION_ID, "--project", PROJECT_PATH], check=True)

    gemini_dir = Path(os.path.expanduser("~")) / ".gemini" / "tmp" / "ml2" / "chats"
    matches = list(gemini_dir.glob(f"*{TEST_SESSION_ID[:8]}*.json"))
    if not matches:
        print("FAILED: Gemini session file not created.")
        return

    with open(matches[0]) as f:
        data = json.load(f)

    msgs = data["messages"]
    print(f"Gemini session: {len(msgs)} messages")

    errors = []

    # Check Read -> read_file mapping with result attached
    a1 = next((m for m in msgs if m.get("id") == "a1"), None)
    if not a1:
        errors.append("FAILED: assistant message a1 not found")
    else:
        tc_read = next((tc for tc in a1["toolCalls"] if tc["name"] == "read_file"), None)
        if not tc_read:
            errors.append("FAILED: Read -> read_file not found")
        elif tc_read.get("status") != "success":
            errors.append(f"FAILED: Read tool result not attached (status={tc_read.get('status')})")
        else:
            print("OK: Read -> read_file, result attached")

        # Check TaskCreate -> task_create
        tc_task = next((tc for tc in a1["toolCalls"] if tc["name"] == "task_create"), None)
        if not tc_task:
            errors.append("FAILED: TaskCreate -> task_create not found")
        elif tc_task["args"].get("subject") != "Test Task":
            errors.append(f"FAILED: task_create subject wrong: {tc_task['args']}")
        else:
            print("OK: TaskCreate -> task_create, args preserved")

        # Check thinking -> thoughts
        if not a1.get("thoughts"):
            errors.append("FAILED: thinking block not converted to thoughts")
        else:
            print("OK: thinking -> thoughts")

        # Check tokens
        if not a1.get("tokens", {}).get("output"):
            errors.append("FAILED: token counts not preserved")
        else:
            print(f"OK: tokens preserved: {a1['tokens']}")

    # Check TaskUpdate -> task_update
    a2 = next((m for m in msgs if m.get("id") == "a2"), None)
    if not a2:
        errors.append("FAILED: assistant message a2 not found")
    else:
        tc_upd = next((tc for tc in a2["toolCalls"] if tc["name"] == "task_update"), None)
        if not tc_upd:
            errors.append("FAILED: TaskUpdate -> task_update not found")
        elif tc_upd["args"].get("status") != "completed":
            errors.append(f"FAILED: task_update status wrong: {tc_upd['args']}")
        else:
            print("OK: TaskUpdate -> task_update, status preserved")

    # Check skipped entries: tool results are embedded in tool calls, not separate messages
    # Expected: 1 user + 2 gemini = 3 total
    if len(msgs) != 3:
        errors.append(f"FAILED: expected 3 messages (1 user + 2 gemini), got {len(msgs)} — metadata entries not skipped?")
    else:
        print("OK: metadata entries (file-history-snapshot, system) skipped, tool results embedded")

    if errors:
        for e in errors:
            print(e)
    else:
        print("ALL CHECKS PASSED")

    # Cleanup
    mock_file.unlink(missing_ok=True)
    for f in matches:
        f.unlink(missing_ok=True)
    print("Cleaned up test files.")
    print("--- Test Complete ---")


if __name__ == "__main__":
    run_test()
