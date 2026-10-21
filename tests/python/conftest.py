"""Register the device option used by framework UT; leave legacy tests unchanged."""


def pytest_addoption(parser):
    group = parser.getgroup("framework NPU tests")
    group.addoption(
        "--npu-device",
        type=int,
        default=None,
        help="Visible logical NPU index for framework UT; omitted keeps the current NPU.",
    )
