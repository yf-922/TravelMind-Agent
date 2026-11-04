from scripts.launch_checkpoint_b1 import openai_messages


def test_langchain_human_role_is_openai_user():
    messages = [{"role": "system", "content": "s"}, {"role": "human", "content": "u"}]
    assert openai_messages(messages) == [
        {"role": "system", "content": "s"}, {"role": "user", "content": "u"}
    ]


def test_existing_openai_roles_are_preserved():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    assert openai_messages(messages) == messages
