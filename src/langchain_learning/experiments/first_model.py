"""First experiment: send a prompt to the configured chat model."""

from langchain_learning.models import get_chat_model


def main() -> None:
    """Invoke the configured model and print its response."""
    model = get_chat_model(temperature=1)
    response = model.invoke(
        "Invent a new holiday. Give it a name and describe how people celebrate it in three sentences"
    )
    print(response.content)


if __name__ == "__main__":
    main()
