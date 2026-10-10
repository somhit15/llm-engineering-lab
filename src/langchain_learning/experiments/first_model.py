"""First experiment: send a prompt to the configured chat model."""

from langchain_learning.models import get_chat_model


def main() -> None:
    """Invoke the configured model and print its response."""
    model = get_chat_model()
    response = model.invoke(
        "Explain what LangChain is in three concise sentences."
    )
    print(response.content)


if __name__ == "__main__":
    main()
