"""Per-room persistent memory with sliding context window, TTL, and decay."""
import time
import json
import os
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

@dataclass
class MemoryEntry:
    content: str
    room: str = ""
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5
    ttl: float = 0.0  # 0 = never expires
    id: str = ""
    access_count: int = 0

class RoomMemory:
    def __init__(self, max_per_room: int = 100, context_window: int = 10,
                 decay_rate: float = 0.999, persist_path: str = ""):
        self._rooms: dict[str, deque] = {}
        self.max_per_room = max_per_room
        self.context_window = context_window
        self.decay_rate = decay_rate
        self.persist_path = persist_path
        self._total_stored = 0

    def store(self, room: str, content: str, tags: list[str] = None,
              importance: float = 0.5, ttl: float = 0.0) -> MemoryEntry:
        if room not in self._rooms:
            self._rooms[room] = deque(maxlen=self.max_per_room)
        entry = MemoryEntry(content=content, room=room, tags=tags or [],
                          importance=importance, ttl=ttl,
                          id=f"{room}-{self._total_stored}")
        self._rooms[room].append(entry)
        self._total_stored += 1
        return entry

    def store_batch(self, room: str, entries: list[dict]) -> list[MemoryEntry]:
        return [self.store(room, e.get("content", ""), e.get("tags"),
                          e.get("importance", 0.5), e.get("ttl", 0.0)) for e in entries]

    def recall(self, room: str, n: int = 0) -> list[MemoryEntry]:
        entries = list(self._rooms.get(room, []))
        now = time.time()
        # Filter expired
        entries = [e for e in entries if e.ttl == 0 or now - e.timestamp < e.ttl]
        # Apply importance decay
        for e in entries:
            e.importance *= self.decay_rate
        if n > 0:
            return entries[-n:]
        return entries[-self.context_window:]

    def recall_important(self, room: str, n: int = 5) -> list[MemoryEntry]:
        entries = list(self._rooms.get(room, []))
        now = time.time()
        entries = [e for e in entries if e.ttl == 0 or now - e.timestamp < e.ttl]
        entries.sort(key=lambda e: e.importance, reverse=True)
        return entries[:n]

    def search(self, query: str, room: str = "", limit: int = 20) -> list[MemoryEntry]:
        q = query.lower()
        results = []
        rooms = {room} if room else set(self._rooms.keys())
        now = time.time()
        for r in rooms:
            for entry in self._rooms.get(r, []):
                if entry.ttl > 0 and now - entry.timestamp >= entry.ttl:
                    continue
                if q in entry.content.lower() or q in " ".join(entry.tags).lower():
                    entry.access_count += 1
                    results.append(entry)
        results.sort(key=lambda e: e.importance, reverse=True)
        return results[:limit]

    def forget_room(self, room: str) -> int:
        entries = self._rooms.pop(room, None)
        return len(entries) if entries else 0

    def forget_entry(self, room: str, entry_id: str) -> bool:
        entries = self._rooms.get(room)
        if not entries:
            return False
        for i, e in enumerate(entries):
            if e.id == entry_id:
                entries.remove(e)
                return True
        return False

    def purge_expired(self) -> int:
        now = time.time()
        purged = 0
        for room_id in list(self._rooms.keys()):
            entries = self._rooms[room_id]
            before = len(entries)
            self._rooms[room_id] = deque(
                [e for e in entries if e.ttl == 0 or now - e.timestamp < e.ttl],
                maxlen=self.max_per_room)
            purged += before - len(self._rooms[room_id])
        return purged

    def room_names(self) -> list[str]:
        return list(self._rooms.keys())

    def save(self, path: str = ""):
        path = path or self.persist_path
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {}
        for room, entries in self._rooms.items():
            data[room] = [{"content": e.content, "tags": e.tags, "importance": e.importance,
                          "ttl": e.ttl, "timestamp": e.timestamp, "id": e.id}
                         for e in entries]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str = ""):
        path = path or self.persist_path
        if not path or not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        for room, entries in data.items():
            for e in entries:
                entry = MemoryEntry(content=e["content"], room=room,
                                  tags=e.get("tags", []), importance=e.get("importance", 0.5),
                                  ttl=e.get("ttl", 0.0), timestamp=e.get("timestamp", time.time()),
                                  id=e.get("id", ""))
                if room not in self._rooms:
                    self._rooms[room] = deque(maxlen=self.max_per_room)
                self._rooms[room].append(entry)

    @property
    def stats(self) -> dict:
        return {"rooms": len(self._rooms),
                "total_entries": sum(len(v) for v in self._rooms.values()),
                "total_stored": self._total_stored,
                "context_window": self.context_window,
                "decay_rate": self.decay_rate}
