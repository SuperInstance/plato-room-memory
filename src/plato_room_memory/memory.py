"""Per-room persistent memory with sliding context window."""
import time
from dataclasses import dataclass, field
from collections import deque

@dataclass
class MemoryEntry:
    content: str
    room: str = ""
    timestamp: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5

class RoomMemory:
    def __init__(self, max_per_room: int = 100, context_window: int = 10):
        self._rooms: dict[str, deque] = {}
        self.max_per_room = max_per_room
        self.context_window = context_window

    def store(self, room: str, content: str, tags: list[str] = None, importance: float = 0.5) -> MemoryEntry:
        if room not in self._rooms:
            self._rooms[room] = deque(maxlen=self.max_per_room)
        entry = MemoryEntry(content=content, room=room, tags=tags or [], importance=importance)
        self._rooms[room].append(entry)
        return entry

    def recall(self, room: str, n: int = 0) -> list[MemoryEntry]:
        entries = list(self._rooms.get(room, []))
        if n > 0:
            return entries[-n:]
        return entries[-self.context_window:]

    def search(self, query: str, room: str = "") -> list[MemoryEntry]:
        q = query.lower()
        results = []
        rooms = {room} if room else set(self._rooms.keys())
        for r in rooms:
            for entry in self._rooms.get(r, []):
                if q in entry.content.lower() or q in " ".join(entry.tags).lower():
                    results.append(entry)
        results.sort(key=lambda e: e.importance, reverse=True)
        return results

    def forget_room(self, room: str):
        self._rooms.pop(room, None)

    @property
    def stats(self) -> dict:
        return {"rooms": len(self._rooms),
                "total_entries": sum(len(v) for v in self._rooms.values()),
                "context_window": self.context_window}
