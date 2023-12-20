import blade as bl

if __name__ == "__main__":

    host_input = bl.tensor((5), dtype=bl.f32, device=bl.cpu)

    print(host_input)

    assert host_input.unified == False
    assert host_input.hash != 0

    assert host_input.shape[0] == 5