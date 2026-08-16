from mammography_agent.datasets.manager import selected

def test_one_dataset_does_not_expand(): assert selected(["cbis_ddsm"])==["cbis_ddsm"]
def test_all_expands_configured(): assert set(selected(["all"]))=={"cbis_ddsm","cmmd","vindr"}
