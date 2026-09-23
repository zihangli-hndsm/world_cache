import torch


def test_torch_cuda_is_available() -> None:
    assert torch.cuda.is_available()


def test_habitat_sim_imports() -> None:
    import habitat_sim

    assert habitat_sim is not None

