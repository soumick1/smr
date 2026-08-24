import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: needs the GPU server, real weights and SMR_GPU_TESTS=1")
