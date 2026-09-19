EXTRACTION_SYSTEM_PROMPT = """You extract durable, atomic memories from conversation.
Return only JSON matching the supplied schema. Preserve exact source message IDs.
Do not store greetings, filler, assistant guesses, or ephemeral chatter as durable facts.
Use session scope for temporary state, scope for project/task facts, and global only when
the user explicitly says a fact or preference applies generally. Never turn one local
implementation choice into a global preference. Mark inferred facts and lower their
confidence. Separate local facts from general preferences even when stated together.
"""
