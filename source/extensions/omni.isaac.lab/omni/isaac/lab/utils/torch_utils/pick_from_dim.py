import torch

def pick_from_dim(input : torch.Tensor, foreach_in_this_dim : int, pick_from_this_dim : int, pick_idx : torch.Tensor):
    """ Wrapper around gather for simpler case.
    if input is shape([L, M, N]), pick_idx is shape [L], foreach in dim 0, pick from dim 1,
     the result is shape([L, N]), equivalent to 
      result[i] = input[i, pick_idx[i], :] """
    assert pick_idx.dim() == 1
    P = pick_idx.size(0) # example: L
    # gather index shape pre expand: example [L, 1, 1]
    # all dimensions 1 except for the foreach_in_this_dim
    gather_idx_shape_pre_expand = [P if i == foreach_in_this_dim else 1 for i in range(input.dim())]
    gather_idx_pre_expand = pick_idx.view(gather_idx_shape_pre_expand) # example [L, 1, 1]
    # gather index shape (expanded): example [L, 1, N]
    # all dimensions same as input except for the foreach_in_this_dim (size = P) and pick_from_dim (size = 1)
    gather_idx_shape = [(g if (i == foreach_in_this_dim or i == pick_from_this_dim) else input.shape[i]) for i, g in enumerate(gather_idx_shape_pre_expand)] # example [L, 1, N]
    gather_idx = gather_idx_pre_expand.expand(gather_idx_shape) # example [L, 1, N]
    result = input.gather(pick_from_this_dim, gather_idx).squeeze(pick_from_this_dim) # example [L, N]
    return result
    
    
def test_pick_from_dim():
    # input is a (100, 4, 9) tensor full of random values
    input = torch.rand(100, 4, 9)
    pick_idx = torch.randint(4, (100,))
    out = pick_from_dim(input, 0, 1, pick_idx)
    assert out.shape == (100, 9)
    for i in range(100):
        # assert out[i] == input[i, pick_idx[i], :]
        assert torch.allclose(out[i], input[i, pick_idx[i], :])
    # same for (4, 100, 9) tensor
    input = torch.rand(4, 100, 9)
    pick_idx = torch.randint(4, (100,))
    out = pick_from_dim(input, 1, 0, pick_idx)
    assert out.shape == (100, 9)
    for i in range(100):
        # assert out[i] == input[pick_idx[i], i, :]
        assert torch.allclose(out[i], input[pick_idx[i], i, :])
    # same for (4, 100, 9) tensor
    # same for (100, 9, 4) tensor
    input = torch.rand(100, 9, 4)
    pick_idx = torch.randint(4, (100,))
    out = pick_from_dim(input, 0, 2, pick_idx)
    assert out.shape == (100, 9)
    for i in range(100):
        # assert out[i] == input[i, :, pick_idx[i]]
        assert torch.allclose(out[i], input[i, :, pick_idx[i]])
    print("pick_from_dim passed")

if __name__ == "__main__":
    test_pick_from_dim()