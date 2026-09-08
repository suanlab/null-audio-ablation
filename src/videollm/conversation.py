"""Chat template definitions for VideoLLM conversations."""

from dataclasses import dataclass, field


@dataclass
class Conversation:
    """A conversation template for multi-turn video/audio chat."""

    system: str = ""
    roles: tuple[str, str] = ("user", "assistant")
    messages: list[list[str]] = field(default_factory=list)
    sep: str = "\n"
    sep2: str = ""

    def get_prompt(self) -> str:
        """Build the full prompt string from conversation history."""
        parts: list[str] = []
        if self.system:
            parts.append(self.system + self.sep)

        for _i, (role_content) in enumerate(self.messages):
            role = role_content[0]
            content = role_content[1] if len(role_content) > 1 else ""
            if content:
                parts.append(f"{role}: {content}{self.sep}")
            else:
                parts.append(f"{role}:")
        return "".join(parts)

    def append_message(self, role: str, message: str) -> None:
        """Append a message to the conversation."""
        self.messages.append([role, message])

    def copy(self) -> "Conversation":
        """Return a deep copy of this conversation."""
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[list(m) for m in self.messages],
            sep=self.sep,
            sep2=self.sep2,
        )


# --- Pre-defined templates ---

CONV_QWEN2 = Conversation(
    system=(
        "You are a helpful assistant that can understand videos and audio. "
        "Analyze the visual and audio content carefully before answering."
    ),
    roles=("user", "assistant"),
    sep="\n",
)

CONV_PLAIN = Conversation(
    system="",
    roles=("", ""),
    sep="\n",
)

CONV_TEMPLATES: dict[str, Conversation] = {
    "qwen2": CONV_QWEN2,
    "plain": CONV_PLAIN,
}


def get_conversation_template(name: str) -> Conversation:
    """Get a copy of a named conversation template.

    Args:
        name: Template name (e.g. 'qwen2', 'plain').

    Returns:
        A fresh copy of the conversation template.

    Raises:
        KeyError: If the template name is not found.
    """
    if name not in CONV_TEMPLATES:
        raise KeyError(f"Unknown conversation template '{name}'. Available: {list(CONV_TEMPLATES.keys())}")
    return CONV_TEMPLATES[name].copy()
