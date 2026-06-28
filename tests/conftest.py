from rdkit import RDLogger


def pytest_configure(config):
    RDLogger.DisableLog("rdApp.warning")
