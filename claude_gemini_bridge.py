import json
import os
import uuid
import argparse
from datetime import datetime, timezone
from pathlib import Path

def safe_int(v, default=0):
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# ==============================================================================
# TOOL MAPPING CONFIGURATION
# ==============================================================================

# Claude -> Gemini
C2G_MAP = {
    "Bash":       {"name": "run_shell_command", "args": lambda x: {"command": x.get("command"), "description": x.get("description", "")}},
    "Read":       {"name": "read_file",         "args": lambda x: {"file_path": x.get("file_path") or x.get("path"), "start_line": safe_int(x.get("offset")), "end_line": (safe_int(x.get("offset")) + safe_int(x.get("limit"))) or None}},
    "Write":      {"name": "write_file",        "args": lambda x: {"file_path": x.get("file_path") or x.get("path"), "content": x.get("content")}},
    "Edit":       {"name": "replace",           "args": lambda x: {"file_path": x.get("file_path") or x.get("path"), "old_string": x.get("old_string"), "new_string": x.get("new_string")}},
    "Glob":       {"name": "list_directory",    "args": lambda x: {"dir_path": x.get("path", "."), "pattern": x.get("pattern")}},
    "Grep":       {"name": "grep_search",       "args": lambda x: {"pattern": x.get("pattern"), "dir_path": x.get("path", ".")}},
    "Agent":      {"name": "generalist",        "args": lambda x: {"request": x.get("prompt")}},
    "ToolSearch": {"name": "tool_search",       "args": lambda x: {"query": x.get("query")}},
    "TaskCreate": {"name": "task_create",       "args": lambda x: {"subject": x.get("subject"), "description": x.get("description"), "active_form": x.get("activeForm")}},
    "TaskUpdate": {"name": "task_update",       "args": lambda x: {"task_id": x.get("taskId"), "status": x.get("status"), "subject": x.get("subject"), "description": x.get("description")}},
    "TaskOutput": {"name": "task_output",       "args": lambda x: {"task_id": x.get("taskId")}},
    "TaskStop":   {"name": "task_stop",         "args": lambda x: {"task_id": x.get("taskId")}},
}

# Gemini -> Claude
G2C_MAP = {
    "run_shell_command": {"name": "Bash",       "args": lambda x: {"command": x.get("command")}},
    "read_file":         {"name": "Read",       "args": lambda x: {"file_path": x.get("file_path"), "offset": x.get("start_line"), "limit": (x.get("end_line") - x.get("start_line", 0)) if x.get("end_line") else None}},
    "write_file":        {"name": "Write",      "args": lambda x: {"file_path": x.get("file_path"), "content": x.get("content")}},
    "replace":           {"name": "Edit",       "args": lambda x: {"file_path": x.get("file_path"), "old_string": x.get("old_string"), "new_string": x.get("new_string")}},
    "list_directory":    {"name": "Glob",       "args": lambda x: {"path": x.get("dir_path"), "pattern": x.get("pattern", "**/*")}},
    "grep_search":       {"name": "Grep",       "args": lambda x: {"pattern": x.get("pattern"), "path": x.get("dir_path")}},
    "generalist":        {"name": "Agent",      "args": lambda x: {"prompt": x.get("request")}},
    "tool_search":       {"name": "ToolSearch", "args": lambda x: {"query": x.get("query")}},
    "task_create":       {"name": "TaskCreate", "args": lambda x: {"subject": x.get("subject"), "description": x.get("description"), "activeForm": x.get("active_form")}},
    "task_update":       {"name": "TaskUpdate", "args": lambda x: {"taskId": x.get("task_id"), "status": x.get("status"), "subject": x.get("subject"), "description": x.get("description")}},
    "task_output":       {"name": "TaskOutput", "args": lambda x: {"taskId": x.get("task_id")}},
    "task_stop":         {"name": "TaskStop",   "args": lambda x: {"taskId": x.get("task_id")}},
}

# Entry types to skip entirely (Claude-internal metadata)
CLAUDE_SKIP_TYPES = {"file-history-snapshot", "system", "last-prompt", "queue-operation"}

# ==============================================================================
# HELPERS
# ==============================================================================

HOME = Path(os.path.expanduser("~"))
CLAUDE_DIR = HOME / ".claude"
GEMINI_DIR = HOME / ".gemini"


def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_gemini_project_hash(project_path):
    project_path = project_path.rstrip("/")
    projects_file = GEMINI_DIR / "projects.json"
    if not projects_file.exists():
        return "default"
    with open(projects_file) as f:
        data = json.load(f)
    return data.get("projects", {}).get(project_path, "default")


def map_tool_call(tool_name, args, mapping):
    if tool_name in mapping:
        entry = mapping[tool_name]
        try:
            mapped_args = {k: v for k, v in entry["args"](args or {}).items() if v is not None}
            return entry["name"], mapped_args
        except Exception as e:
            print(f"Warning: Failed to map args for {tool_name}: {e}")
    return tool_name, args or {}


def extract_progress_text(data):
    """Convert a Claude 'progress' entry (subagent) into a readable text block."""
    inner = data.get("data", {})
    msg = inner.get("message", {})
    content = msg.get("message", {}).get("content", [])
    parts = []
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c["text"])
    elif isinstance(content, str):
        parts.append(content)
    text = "\n".join(parts).strip()
    if text:
        return f"[Subagent] {text[:500]}"
    return None


# ==============================================================================
# CLAUDE -> GEMINI
# ==============================================================================

def migrate_claude_to_gemini(session_id, project_path):
    project_slug = project_path.strip("/").replace("/", "-")
    proj_dir = CLAUDE_DIR / "projects" / f"-{project_slug}"
    if not proj_dir.exists():
        proj_dir = CLAUDE_DIR / "projects" / project_slug
    
    if not proj_dir.exists():
        print(f"Error: Project directory not found at {proj_dir}")
        print("Available projects:")
        for d in (CLAUDE_DIR / "projects").iterdir():
            if d.is_dir():
                print(f"  {d.name}")
        return

    source_path = proj_dir / f"{session_id}.jsonl"
    if not source_path.exists():
        # Try finding by prefix
        matches = list(proj_dir.glob(f"*{session_id}*.jsonl"))
        if matches:
            source_path = matches[0]
            session_id = source_path.stem
            print(f"Found session by match: {session_id}")
        else:
            print(f"Error: Claude session {session_id} not found in {proj_dir}")
            print("Available sessions:")
            for f in proj_dir.glob("*.jsonl"):
                print(f"  {f.stem}")
            return

    project_hash = get_gemini_project_hash(project_path)
    target_dir = GEMINI_DIR / "tmp" / project_hash / "chats"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"session-{datetime.now().strftime('%Y-%m-%dT%H-%M')}-{session_id[:8]}.json"

    messages = []
    tool_call_map = {}  # claude tool_use id -> (msg_index, tool_call_index)

    with open(source_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entry_type = data.get("type", "")

            if entry_type in CLAUDE_SKIP_TYPES:
                continue

            elif entry_type == "progress":
                # Subagent output — append as a text annotation to the last gemini message
                text = extract_progress_text(data)
                if text and messages and messages[-1]["type"] == "gemini":
                    messages[-1]["content"] += f"\n\n{text}"

            elif entry_type == "user":
                content = data["message"].get("content", [])
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]

                tool_results = [c for c in content if isinstance(c, dict) and c.get("type") == "tool_result"]
                text_parts   = [c for c in content if isinstance(c, dict) and c.get("type") == "text"]

                # Attach tool results back to their tool call
                for res in tool_results:
                    tid = res.get("tool_use_id", "")
                    if tid in tool_call_map:
                        m_idx, c_idx = tool_call_map[tid]
                        tc = messages[m_idx]["toolCalls"][c_idx]
                        tc["status"] = "error" if res.get("is_error") else "success"
                        raw = res.get("content", "")
                        # content can be a list of parts
                        if isinstance(raw, list):
                            raw = "\n".join(p.get("text", str(p)) for p in raw)
                        tc["result"] = [{"functionResponse": {"id": tid, "name": tc["name"],
                                          "response": {"output": raw, "exitCode": 1 if res.get("is_error") else 0}}}]
                        tc["resultDisplay"] = str(raw)[:2000]

                if not tool_results and text_parts:
                    msg = {
                        "id": data.get("uuid", str(uuid.uuid4())),
                        "timestamp": data["timestamp"],
                        "type": "user",
                        "content": [{"text": c["text"]} for c in text_parts],
                    }
                    messages.append(msg)

            elif entry_type == "assistant":
                raw_msg = data.get("message", {})
                usage = raw_msg.get("usage", {})
                msg = {
                    "id": data.get("uuid", str(uuid.uuid4())),
                    "timestamp": data["timestamp"],
                    "type": "gemini",
                    "content": "",
                    "thoughts": [],
                    "toolCalls": [],
                    "model": raw_msg.get("model", "claude-sonnet-4-6"),
                    "tokens": {
                        "input":  usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0),
                        "output": usage.get("output_tokens", 0),
                        "cached": usage.get("cache_read_input_tokens", 0),
                        "total":  usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                    },
                }

                for part in raw_msg.get("content", []):
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype == "thinking":
                        msg["thoughts"].append({
                            "subject": "Thinking",
                            "description": part.get("thinking", ""),
                            "timestamp": data["timestamp"],
                        })
                    elif ptype == "text":
                        msg["content"] += part.get("text", "")
                    elif ptype == "tool_use":
                        t_name, t_args = map_tool_call(part["name"], part.get("input"), C2G_MAP)
                        tc = {
                            "id": part["id"],
                            "name": t_name,
                            "args": t_args,
                            "status": "pending",
                            "timestamp": data["timestamp"],
                            "displayName": part["name"],  # keep original for readability
                        }
                        tool_call_map[part["id"]] = (len(messages), len(msg["toolCalls"]))
                        msg["toolCalls"].append(tc)

                messages.append(msg)

    gemini_session = {
        "sessionId": session_id,
        "projectHash": project_hash,
        "startTime": messages[0]["timestamp"] if messages else "",
        "lastUpdated": now_iso(),
        "messages": messages,
    }
    with open(target_path, "w") as f:
        json.dump(gemini_session, f, indent=2)
    print(f"Migrated Claude -> Gemini: {target_path}")
    print(f"To resume: cd {project_path} && gemini --resume {session_id}")


# ==============================================================================
# GEMINI -> CLAUDE
# ==============================================================================

def migrate_gemini_to_claude(session_id, project_path):
    project_hash = get_gemini_project_hash(project_path)
    source_dir = GEMINI_DIR / "tmp" / project_hash / "chats"
    matches = list(source_dir.glob(f"*{session_id[:8]}*.json"))
    if not matches:
        print(f"Error: Gemini session {session_id} not found in {source_dir}")
        return
    source_path = sorted(matches, key=os.path.getmtime)[-1]

    project_slug = project_path.strip("/").replace("/", "-")
    target_dir = CLAUDE_DIR / "projects" / f"-{project_slug}"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{session_id}.jsonl"

    with open(source_path) as f:
        gemini_data = json.load(f)

    def make_claude_base(msg_id, ts):
        return {"uuid": msg_id, "timestamp": ts, "sessionId": session_id, "cwd": project_path,
                "parentUuid": None, "isSidechain": False, "userType": "external", "version": "0.0.0"}

    with open(target_path, "w") as f:
        for msg in gemini_data.get("messages", []):
            ts = msg.get("timestamp", now_iso())
            mid = msg.get("id", str(uuid.uuid4()))

            if msg["type"] == "user":
                entry = {**make_claude_base(mid, ts), "type": "user",
                         "message": {"role": "user",
                                     "content": [{"type": "text", "text": c["text"]} for c in msg.get("content", []) if "text" in c]}}
                f.write(json.dumps(entry) + "\n")

            elif msg["type"] == "gemini":
                claude_content = []

                # Thoughts -> thinking blocks
                for th in msg.get("thoughts", []):
                    claude_content.append({"type": "thinking", "thinking": th.get("description", "")})

                # Main text
                if msg.get("content"):
                    claude_content.append({"type": "text", "text": msg["content"]})

                # Tool calls (collect results to emit as separate user messages)
                tool_result_msgs = []
                for tc in msg.get("toolCalls", []):
                    t_name, t_args = map_tool_call(tc["name"], tc.get("args"), G2C_MAP)
                    tid = tc.get("id", str(uuid.uuid4()))
                    claude_content.append({"type": "tool_use", "id": tid, "name": t_name, "input": t_args})

                    res_val = ""
                    is_error = tc.get("status") == "error"
                    if tc.get("result"):
                        resp = tc["result"][0].get("functionResponse", {}).get("response", {})
                        res_val = resp.get("output", resp.get("error", ""))

                    res_entry = {
                        **make_claude_base(str(uuid.uuid4()), tc.get("timestamp", ts)),
                        "type": "user",
                        "message": {"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": tid, "content": res_val, "is_error": is_error}
                        ]},
                    }
                    tool_result_msgs.append(res_entry)

                assistant_entry = {
                    **make_claude_base(mid, ts),
                    "type": "assistant",
                    "message": {"role": "assistant", "content": claude_content, "model": msg.get("model", "")},
                }
                f.write(json.dumps(assistant_entry) + "\n")
                for res_entry in tool_result_msgs:
                    f.write(json.dumps(res_entry) + "\n")

    print(f"Migrated Gemini -> Claude: {target_path}")
    print(f"To resume: cd {project_path} && claude --resume {session_id}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--import-cli", choices=["claude", "gemini"], required=True)
    parser.add_argument("-e", "--export-cli", choices=["claude", "gemini"], required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--project", default=os.getcwd())
    args = parser.parse_args()

    if args.import_cli == "claude" and args.export_cli == "gemini":
        migrate_claude_to_gemini(args.id, args.project)
    elif args.import_cli == "gemini" and args.export_cli == "claude":
        migrate_gemini_to_claude(args.id, args.project)
    else:
        print("Error: --import-cli and --export-cli must differ.")
