import torch

from model.gpt import GPT


class GPTGenerator:
    """
    Generate text using a trained GPT model.
    """

    def __init__(
        self,
        model: GPT,
        tokenizer,
    ) -> None:

        self.model = model
        self.tokenizer = tokenizer
        self.device = model.config.device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> str:

        self.model.eval()

        # Encode prompt
        token_ids = self.tokenizer.encode(
            prompt,
            out_type=int,
        )

        tokens = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():

            for _ in range(max_new_tokens):

                # Keep only the latest context
                tokens = tokens[
                    :,
                    -self.model.config.max_sequence_length:
                ]

                logits = self.model(tokens)

                # Next token logits
                logits = logits[:, -1, :]

                # Greedy decoding
                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True,
                )

                tokens = torch.cat(
                    (tokens, next_token),
                    dim=1,
                )

        generated = tokens[0].tolist()

        return self.tokenizer.decode(generated)