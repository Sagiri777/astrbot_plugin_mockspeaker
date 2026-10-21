from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    allowed_groups: tuple[str, ...] = ()
    min_messages: int = 100
    update_every: int = 50
    max_samples: int = 300
    max_message_chars: int = 500
    analysis_chars: int = 40000
    min_message_chars: int = 2
    provider_id: str = ""
    summary_every: int = 6
    context_turns: int = 20
    summary_chars: int = 1600
    timeout: int = 90
    sync_personas: bool = False
    cross_group_profiles: bool = False
    excluded_users: tuple[str, ...] = ()
    command_prefixes: tuple[str, ...] = ("/",)

    @classmethod
    def from_config(cls, config):
        values = {}
        limits = {
            "min_messages": (1, 1000000),
            "update_every": (1, 1000000),
            "max_samples": (10, 1000),
            "max_message_chars": (20, 2000),
            "analysis_chars": (2000, 200000),
            "min_message_chars": (1, 100),
            "summary_every": (1, 20),
            "context_turns": (2, 50),
            "summary_chars": (200, 4000),
            "timeout": (5, 300),
        }
        defaults = cls()
        for key, (low, high) in limits.items():
            value = int(config.get(key, getattr(defaults, key)))
            if not low <= value <= high:
                raise ValueError(f"{key} 必须在 {low} 到 {high} 之间")
            values[key] = value
        for key in ("allowed_groups", "excluded_users", "command_prefixes"):
            values[key] = tuple(str(x).strip() for x in config.get(key, getattr(defaults, key)) if str(x).strip())
        values["provider_id"] = str(config.get("provider_id", "")).strip()
        values["sync_personas"] = bool(config.get("sync_personas", False))
        values["cross_group_profiles"] = bool(config.get("cross_group_profiles", False))
        if values["summary_every"] > values["context_turns"]:
            raise ValueError("summary_every 不能大于 context_turns")
        return cls(**values)

    def allows(self, platform: str, group: str) -> bool:
        return bool(group) and (group in self.allowed_groups or f"{platform}:{group}" in self.allowed_groups)

    def groups_for(self, platform: str) -> tuple[str, ...]:
        prefix = platform + ":"
        return tuple(
            sorted(
                {
                    entry[len(prefix) :] if entry.startswith(prefix) else entry
                    for entry in self.allowed_groups
                    if ":" not in entry or entry.startswith(prefix)
                }
            )
        )
