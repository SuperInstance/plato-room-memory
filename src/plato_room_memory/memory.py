"""Room memory — stores room memories with decay, importance scoring, consolidation, and search."""
import time
import math
import re
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import defaultdict
from enum import Enum

class MemoryType(Enum):
    FACT = "fact"
    EVENT = "event"
    PREFERENCE = "preference"
    DECISION = "decision"
    CONTEXT = "context"
    EPHEMERAL = "ephemeral"

@dataclass
class Memory:
    id: str
    content: str
    memory_type: MemoryType = MemoryType.FACT
    room: str = ""
    source: str = ""
    importance: float = 0.5  # 0.0-1.0
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = 0.0
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)
    decay_rate: float = 0.99  # per hour

@dataclass
class ConsolidatedMemory:
    id: str
    content: str
    source_ids: list[str]
    room: str = ""
    importance: float = 0.0
    created_at: float = field(default_factory=time.time)

class RoomMemory:
    def __init__(self, decay_enabled: bool = True, consolidation_threshold: int = 100):
        self.decay_enabled = decay_enabled
        self.consolidation_threshold = consolidation_threshold
        self._memories: dict[str, dict[str, Memory]] = defaultdict(dict)  # room → {id → memory}
        self._consolidated: dict[str, list[ConsolidatedMemory]] = defaultdict(list)
        self._decay_log: list[dict] = []

    def remember(self, room: str, content: str, memory_type: str = "fact",
                source: str = "", importance: float = 0.5, tags: list[str] = None,
                confidence: float = 1.0) -> Memory:
        mem_id = hashlib.md5(f"{room}:{content}:{time.time()}".encode()).hexdigest()[:12]
        mem = Memory(id=mem_id, content=content, memory_type=MemoryType(memory_type),
                   room=room, source=source, importance=importance, tags=tags or [],
                   confidence=confidence)
        self._memories[room][mem_id] = mem
        return mem

    def recall(self, room: str, memory_id: str) -> Optional[Memory]:
        mem = self._memories[room].get(memory_id)
        if mem:
            mem.access_count += 1
            mem.last_accessed = time.time()
            # Access boosts importance slightly
            mem.importance = min(1.0, mem.importance + 0.01)
        return mem

    def forget(self, room: str, memory_id: str) -> bool:
        mem = self._memories[room].pop(memory_id, None)
        return mem is not None

    def search(self, room: str, query: str, limit: int = 20) -> list[Memory]:
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        scored = []
        for mem in self._memories[room].values():
            content_lower = mem.content.lower()
            # Simple relevance scoring
            score = 0.0
            if query_lower in content_lower:
                score += 1.0
            for word in query_words:
                if word in content_lower:
                    score += 0.2
            if any(t.lower() in query_lower for t in mem.tags):
                score += 0.5
            score *= mem.importance  # boost by importance
            if score > 0:
                scored.append((mem, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, _ in scored[:limit]]

    def by_type(self, room: str, memory_type: str) -> list[Memory]:
        mt = MemoryType(memory_type)
        return [m for m in self._memories[room].values() if m.memory_type == mt]

    def by_importance(self, room: str, min_importance: float = 0.5) -> list[Memory]:
        return [m for m in self._memories[room].values() if m.importance >= min_importance]

    def top_memories(self, room: str, n: int = 10) -> list[Memory]:
        mems = list(self._memories[room].values())
        mems.sort(key=lambda m: m.importance * (1 + m.access_count * 0.1), reverse=True)
        return mems[:n]

    def decay(self, room: str = "") -> int:
        """Apply importance decay. Returns number of memories decayed."""
        if not self.decay_enabled:
            return 0
        decayed = 0
        rooms = [room] if room else list(self._memories.keys())
        for r in rooms:
            for mem in self._memories[r].values():
                hours_since_access = (time.time() - mem.last_accessed) / 3600 if mem.last_accessed > 0 else (time.time() - mem.created_at) / 3600
                decay_factor = mem.decay_rate ** hours_since_access
                new_importance = mem.importance * decay_factor
                if abs(new_importance - mem.importance) > 0.001:
                    self._decay_log.append({"memory_id": mem.id, "room": r,
                                          "before": mem.importance, "after": new_importance,
                                          "hours": round(hours_since_access, 1)})
                    mem.importance = new_importance
                    decayed += 1
        return decayed

    def purge_forgotten(self, threshold: float = 0.01) -> int:
        """Remove memories below importance threshold."""
        purged = 0
        for room in self._memories:
            to_remove = [mid for mid, m in self._memories[room].items()
                        if m.importance < threshold]
            for mid in to_remove:
                self._memories[room].pop(mid, None)
                purged += 1
        return purged

    def consolidate(self, room: str) -> list[ConsolidatedMemory]:
        """Consolidate low-importance memories into summaries."""
        mems = list(self._memories[room].values())
        if len(mems) < self.consolidation_threshold:
            return []
        # Group by type and consolidate
        by_type = defaultdict(list)
        for m in mems:
            if m.importance < 0.3:
                by_type[m.memory_type.value].append(m)
        results = []
        for mt, group in by_type.items():
            if len(group) < 5:
                continue
            contents = [m.content for m in group]
            # Simple consolidation: join top contents
            consolidated_content = "; ".join(contents[:10])
            avg_importance = sum(m.importance for m in group) / len(group)
            source_ids = [m.id for m in group]
            cm = ConsolidatedMemory(
                id=f"consolidated-{mt}-{int(time.time())}",
                content=consolidated_content, source_ids=source_ids,
                room=room, importance=avg_importance)
            self._consolidated[room].append(cm)
            # Remove consolidated memories
            for m in group:
                self._memories[room].pop(m.id, None)
            results.append(cm)
        return results

    def stats(self, room: str = "") -> dict:
        rooms = [room] if room else list(self._memories.keys())
        total = sum(len(self._memories[r]) for r in rooms)
        types = defaultdict(int)
        for r in rooms:
            for m in self._memories[r].values():
                types[m.memory_type.value] += 1
        return {"rooms": len(rooms), "memories": total, "types": dict(types),
                "consolidated": sum(len(c) for c in self._consolidated.values()),
                "decay_entries": len(self._decay_log)}

import hashlib
